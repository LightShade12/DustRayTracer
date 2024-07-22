#pragma once
#include "BVHNode.cuh"

#include "Core/Kernel/Shaders/Intersection.cuh"
#include "Core/Kernel/Shaders/Anyhit.cuh"

#include "Core/Scene/SceneData.cuh"
#include "Core/CudaMath/helper_math.cuh"

//data layout tightness
//spatial splits when building
//look into 128bit LD instructions for operations
//use grouped float structs instead of separate floats when possible
//look for more early outs in traversal

//Traversal
__device__ void traverseBVH(const Ray& ray, const int root_node_idx, HitPayload* closest_hitpayload, const SceneData* scenedata) {
	if (root_node_idx < 0) return;//empty scene

	const uint8_t maxStackSize = 64;
	int nodeIdxStack[maxStackSize];
	float nodeHitDistStack[maxStackSize];
	uint8_t stackPtr = 0;

	float current_node_hitdist = FLT_MAX;

	nodeIdxStack[stackPtr] = root_node_idx;
	const BVHNode* stackTopNode = &(scenedata->DeviceBVHNodesBuffer[root_node_idx]);//is this in register?
	nodeHitDistStack[stackPtr++] = stackTopNode->m_BoundingBox.intersect(ray);

	ShortHitPayload workinghitpayload;//only to be written to by primitive proccessing
	float child1_hitdist = -1;
	float child2_hitdist = -1;
	const Triangle* primitive = nullptr;

	while (stackPtr > 0) {
		stackTopNode = &(scenedata->DeviceBVHNodesBuffer[nodeIdxStack[--stackPtr]]);
		current_node_hitdist = nodeHitDistStack[stackPtr];

		//custom ray interval culling
		if (!(ray.interval.surrounds(current_node_hitdist)))continue;//TODO: can put this in triangle looping part to get inner clipping working

		//skip nodes farther than closest triangle; redundant
		if (closest_hitpayload->triangle_idx != -1 && closest_hitpayload->hit_distance < current_node_hitdist)continue;

		closest_hitpayload->color += make_float3(1) * 0.05f;

		//if interior
		if (stackTopNode->primitives_indices_count <= 0)
		{
			child1_hitdist = (scenedata->DeviceBVHNodesBuffer[stackTopNode->left_start_idx]).m_BoundingBox.intersect(ray);
			child2_hitdist = (scenedata->DeviceBVHNodesBuffer[stackTopNode->left_start_idx + 1]).m_BoundingBox.intersect(ray);
			//TODO:implement early cull properly see discord for ref
			if (child1_hitdist > child2_hitdist) {
				if (child1_hitdist >= 0 && child1_hitdist < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = child1_hitdist; nodeIdxStack[stackPtr++] = stackTopNode->left_start_idx; }
				if (child2_hitdist >= 0 && child2_hitdist < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = child2_hitdist; nodeIdxStack[stackPtr++] = stackTopNode->left_start_idx + 1; }
			}
			else {
				if (child2_hitdist >= 0 && child2_hitdist < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = child2_hitdist; nodeIdxStack[stackPtr++] = stackTopNode->left_start_idx + 1; }
				if (child1_hitdist >= 0 && child1_hitdist < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = child1_hitdist; nodeIdxStack[stackPtr++] = stackTopNode->left_start_idx; }
			}
		}
		else
		{
			for (int primIndiceIdx = stackTopNode->left_start_idx;
				primIndiceIdx < stackTopNode->left_start_idx + stackTopNode->primitives_indices_count; primIndiceIdx++) {
				int primIdx = scenedata->DeviceBVHPrimitiveIndicesBuffer[primIndiceIdx];
				primitive = &(scenedata->DevicePrimitivesBuffer[primIdx]);
				workinghitpayload = Intersection(ray, primitive, primIdx);

				if (workinghitpayload.triangle_idx != -1 && workinghitpayload.hit_distance < closest_hitpayload->hit_distance) {
					if (!AnyHit(ray, scenedata, &workinghitpayload))continue;
					closest_hitpayload->hit_distance = workinghitpayload.hit_distance;
					closest_hitpayload->triangle_idx = workinghitpayload.triangle_idx;
					closest_hitpayload->UVW = workinghitpayload.UVW;
				}
			}
		}
	}
}

//apparently do not bother sorting nodes for shadow rays and do an early out
__device__ const int traverseBVH_raytest(const Ray& ray, const int root_node_idx, const SceneData* scenedata) {
	if (root_node_idx < 0) return false;//empty scene

	const uint8_t maxStackSize = 64;//TODO: make this const for all
	int nodeIdxStack[maxStackSize];
	uint8_t stackPtr = 0;

	//float current_node_hitdist = FLT_MAX;//redundant?

	nodeIdxStack[stackPtr++] = root_node_idx;
	const BVHNode* stackTopNode = &(scenedata->DeviceBVHNodesBuffer[root_node_idx]);//is this in register?

	ShortHitPayload workinghitpayload;
	float hit1 = -1;
	float hit2 = -1;
	const Triangle* primitive = nullptr;

	while (stackPtr > 0) {
		stackTopNode = &(scenedata->DeviceBVHNodesBuffer[nodeIdxStack[--stackPtr]]);

		float current_node_hitdist = stackTopNode->m_BoundingBox.intersect(ray);

		//custom ray interval culling
		if (!(ray.interval.surrounds(current_node_hitdist)))continue;//TODO: can put this in triangle looping part to get inner clipping working

		//if (current_node_hitdist > ray.interval.max)continue;

		//if leaf
		if (stackTopNode->primitives_indices_count != 0) {
			for (int primIndiceIdx = stackTopNode->left_start_idx;
				primIndiceIdx < stackTopNode->left_start_idx + stackTopNode->primitives_indices_count; primIndiceIdx++) {
				int primIdx = scenedata->DeviceBVHPrimitiveIndicesBuffer[primIndiceIdx];
				primitive = &(scenedata->DevicePrimitivesBuffer[primIdx]);
				workinghitpayload = Intersection(ray, primitive, primIdx);

				if (workinghitpayload.triangle_idx != -1) {
					if (AnyHit(ray, scenedata, &workinghitpayload)) {
						return workinghitpayload.triangle_idx; // Intersection found, return true immediately
					}
				}
			}
		}
		else {
			hit1 = (scenedata->DeviceBVHNodesBuffer[stackTopNode->left_start_idx]).m_BoundingBox.intersect(ray);
			hit2 = (scenedata->DeviceBVHNodesBuffer[stackTopNode->left_start_idx + 1]).m_BoundingBox.intersect(ray);

			if (hit1 > hit2) {
				if (hit1 >= 0) { nodeIdxStack[stackPtr++] = stackTopNode->left_start_idx; }
				if (hit2 >= 0) { nodeIdxStack[stackPtr++] = stackTopNode->left_start_idx + 1; }
			}
			else {
				if (hit2 >= 0) { nodeIdxStack[stackPtr++] = stackTopNode->left_start_idx + 1; }
				if (hit1 >= 0) { nodeIdxStack[stackPtr++] = stackTopNode->left_start_idx; }
			}
		}
	}

	return false;
}