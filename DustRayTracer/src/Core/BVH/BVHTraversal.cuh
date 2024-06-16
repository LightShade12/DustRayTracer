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
		if (closest_hitpayload->primitiveptr != nullptr && closest_hitpayload->hit_distance < current_node_hitdist)continue;

		closest_hitpayload->color += make_float3(1) * 0.05f;

		if (stackTopNode->m_IsLeaf) {
			for (int primIdx = stackTopNode->primitive_start_idx;
				primIdx < stackTopNode->primitive_start_idx + stackTopNode->primitives_count; primIdx++) {
				primitive = &(scenedata->DevicePrimitivesBuffer[primIdx]);
				//primitive = stackTopNode->dev_primitive_ptrs_buffer[primIdx];
				workinghitpayload = Intersection(ray, primitive);

				if (workinghitpayload.primitiveptr != nullptr && workinghitpayload.hit_distance < closest_hitpayload->hit_distance) {
					if (!AnyHit(ray, scenedata, &workinghitpayload))continue;
					closest_hitpayload->hit_distance = workinghitpayload.hit_distance;
					closest_hitpayload->primitiveptr = workinghitpayload.primitiveptr;
					closest_hitpayload->UVW = workinghitpayload.UVW;
				}
			}
		}
		else {
			child1_hitdist = (scenedata->DeviceBVHNodesBuffer[stackTopNode->dev_child1_idx]).m_BoundingBox.intersect(ray);
			child2_hitdist = (scenedata->DeviceBVHNodesBuffer[stackTopNode->dev_child2_idx]).m_BoundingBox.intersect(ray);
			//TODO:implement early cull properly see discord for ref
			if (child1_hitdist > child2_hitdist) {
				if (child1_hitdist >= 0 && child1_hitdist < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = child1_hitdist; nodeIdxStack[stackPtr++] = stackTopNode->dev_child1_idx; }
				if (child2_hitdist >= 0 && child2_hitdist < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = child2_hitdist; nodeIdxStack[stackPtr++] = stackTopNode->dev_child2_idx; }
			}
			else {
				if (child2_hitdist >= 0 && child2_hitdist < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = child2_hitdist; nodeIdxStack[stackPtr++] = stackTopNode->dev_child2_idx; }
				if (child1_hitdist >= 0 && child1_hitdist < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = child1_hitdist; nodeIdxStack[stackPtr++] = stackTopNode->dev_child1_idx; }
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
			for (int primIdx = stackTopNode->primitive_start_idx;
				primIdx < stackTopNode->primitive_start_idx + stackTopNode->primitives_count; primIdx++) {
				const Triangle* primitive = &(scenedata->DevicePrimitivesBuffer[primIdx]);
				workinghitpayload = Intersection(ray, primitive);

				if (workinghitpayload.primitiveptr != nullptr) {
					if (AnyHit(ray, scenedata, &workinghitpayload)) {
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