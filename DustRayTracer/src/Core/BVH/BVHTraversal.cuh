#pragma once
#include "BVHNode.cuh"
#include "Core/Kernel/Shaders/Intersection.cuh"
#include "Core/Kernel/Shaders/Anyhit.cuh"
#include "Core/CudaMath/helper_math.cuh"

//data layout tightness
//spatial splits when building
//look into 128bit LD instructions for operations
//use grouped float structs instead of separate floats when possible
//look for more early outs in traversal

//Traversal
__device__ void traverseBVH(const Ray& ray, const int root_node_idx, HitPayload* closest_hitpayload, bool& debug, const SceneData* scenedata) {
	if (root_node_idx < 0) return;//empty scene

	const uint8_t maxStackSize = 64;
	int nodeIdxStack[maxStackSize];
	float nodeHitDistStack[maxStackSize];
	uint8_t stackPtr = 0;

	float nodeHitDist = FLT_MAX;

	nodeIdxStack[stackPtr] = root_node_idx;
	const BVHNode* stackTopNode = &(scenedata->DeviceBVHNodesBuffer[root_node_idx]);//is this in register?
	nodeHitDistStack[stackPtr++] = stackTopNode->m_BoundingBox.intersect(ray);

	ShortHitPayload workinghitpayload;//only to be written to by primitive proccessing
	float hit1 = -1;
	float hit2 = -1;
	const Triangle* primitive = nullptr;

	while (stackPtr > 0) {
		stackTopNode = &(scenedata->DeviceBVHNodesBuffer[nodeIdxStack[--stackPtr]]);
		nodeHitDist = nodeHitDistStack[stackPtr];

		//custom ray interval culling
		if (!(ray.interval.surrounds(nodeHitDist)))continue;//TODO: can put this in triangle looping part to get inner clipping working

		//skip nodes farther than closest triangle
		if (closest_hitpayload->primitiveptr != nullptr && closest_hitpayload->hit_distance < nodeHitDist) {
			continue;
		}

		closest_hitpayload->color += make_float3(1) * 0.05f;

		if (stackTopNode->m_IsLeaf) {
			for (int primIdx = 0; primIdx < stackTopNode->primitives_count; primIdx++) {
				primitive = stackTopNode->dev_primitive_ptrs_buffer[primIdx];
				workinghitpayload = Intersection(ray, primitive);

				if (workinghitpayload.primitiveptr != nullptr && workinghitpayload.hit_distance < closest_hitpayload->hit_distance) {
					if (!AnyHit(ray, scenedata,
						primitive, workinghitpayload.hit_distance))continue;
					closest_hitpayload->hit_distance = workinghitpayload.hit_distance;
					closest_hitpayload->primitiveptr = workinghitpayload.primitiveptr;
				}
			}
		}
		else {
			hit1 = (scenedata->DeviceBVHNodesBuffer[stackTopNode->dev_child1_idx]).m_BoundingBox.intersect(ray);
			hit2 = (scenedata->DeviceBVHNodesBuffer[stackTopNode->dev_child2_idx]).m_BoundingBox.intersect(ray);
			//TODO:implement early cull properly see discord for ref
			if (hit1 > hit2) {
				if (hit1 >= 0 && hit1 < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = hit1; nodeIdxStack[stackPtr++] = stackTopNode->dev_child1_idx; }
				if (hit2 >= 0 && hit2 < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = hit2; nodeIdxStack[stackPtr++] = stackTopNode->dev_child2_idx; }
			}
			else {
				if (hit2 >= 0 && hit2 < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = hit2; nodeIdxStack[stackPtr++] = stackTopNode->dev_child2_idx; }
				if (hit1 >= 0 && hit1 < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = hit1; nodeIdxStack[stackPtr++] = stackTopNode->dev_child1_idx; }
			}
		}
	}
}

//apparently do not bother sorting nodes for shadow rays and do an early out
__device__ bool traverseBVH_raytest(const Ray& ray, const int root_node_idx, const SceneData* scenedata) {
	if (root_node_idx < 0) return false;//empty scene

	const uint8_t maxStackSize = 64;
	int nodeIdxStack[maxStackSize];
	uint8_t stackPtr = 0;

	float nodeHitDist = FLT_MAX;//redundant?
	const BVHNode* stackTopNode = nullptr;//in register?
	float hit1 = -1;
	float hit2 = -1;
	ShortHitPayload workinghitpayload;

	nodeIdxStack[stackPtr++] = root_node_idx;

	while (stackPtr > 0) {
		stackTopNode = &(scenedata->DeviceBVHNodesBuffer[nodeIdxStack[--stackPtr]]);

		//if root node; one shot
		if (nodeIdxStack[stackPtr] == root_node_idx) {
			float bbox_hitDist = stackTopNode->m_BoundingBox.intersect(ray);

			if (bbox_hitDist < 0) {
				continue;
			}

			nodeHitDist = bbox_hitDist;
		}

		if (stackTopNode->m_IsLeaf) {
			for (int primIdx = 0; primIdx < stackTopNode->primitives_count; primIdx++) {
				const Triangle* primitive = stackTopNode->dev_primitive_ptrs_buffer[primIdx];
				workinghitpayload = Intersection(ray, primitive);

				if (workinghitpayload.primitiveptr != nullptr) {
					if (AnyHit(ray, scenedata, primitive, workinghitpayload.hit_distance)) {
						return true; // Intersection found, return true immediately
					}
				}
			}
		}
		else {
			hit1 = (scenedata->DeviceBVHNodesBuffer[stackTopNode->dev_child1_idx]).m_BoundingBox.intersect(ray);
			hit2 = (scenedata->DeviceBVHNodesBuffer[stackTopNode->dev_child2_idx]).m_BoundingBox.intersect(ray);

			if (hit1 > hit2) {
				if (hit1 >= 0) { nodeIdxStack[stackPtr++] = stackTopNode->dev_child1_idx; }
				if (hit2 >= 0) { nodeIdxStack[stackPtr++] = stackTopNode->dev_child2_idx; }
			}
			else {
				if (hit2 >= 0) { nodeIdxStack[stackPtr++] = stackTopNode->dev_child2_idx; }
				if (hit1 >= 0) { nodeIdxStack[stackPtr++] = stackTopNode->dev_child1_idx; }
			}
		}
	}

	return false;
}