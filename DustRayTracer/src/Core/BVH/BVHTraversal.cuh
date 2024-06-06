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

__device__ HitPayload intersectAABB(const Ray& ray, const Bounds3f& bbox) {
	HitPayload payload;

	//float3 invDir = 1.0f / ray.direction;
	float3 t0 = (bbox.pMin - ray.origin) * ray.invDir;
	float3 t1 = (bbox.pMax - ray.origin) * ray.invDir;

	float3 tmin = fminf(t0, t1);
	float3 tmax = fmaxf(t1, t0);//switched order of t to guard NaNs

	//min max componenet
	float tenter = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
	float texit = fminf(fminf(tmax.x, tmax.y), tmax.z);

	// Adjust tenter if the ray starts inside the AABB
	if (tenter < 0.0f) {
		tenter = 0.0f;
	}

	if (tenter > texit || texit < 0) {
		return payload; // No intersection
	}

	payload.hit_distance = tenter;
	return payload;
}
//Traversal
__device__ void traverseBVH(const Ray& ray, const int root_node_idx, HitPayload* closest_hitpayload, bool& debug, const SceneData* scenedata) {
	if (root_node_idx < 0) return;//empty scene

	const uint8_t maxStackSize = 64; // Adjust based on expected max depth; depends on width of level
	int nodeIdxStack[maxStackSize];
	float nodeHitDistStack[maxStackSize];
	uint8_t stackPtr = 0;

	float nodeHitDist = FLT_MAX;
	const BVHNode* stackTopNode = nullptr;//is this in register?

	nodeHitDistStack[stackPtr] = 0;
	nodeIdxStack[stackPtr++] = root_node_idx;

	while (stackPtr > 0) {
		stackTopNode = &(scenedata->DeviceBVHNodesBuffer[nodeIdxStack[--stackPtr]]);
		nodeHitDist = nodeHitDistStack[stackPtr];

		HitPayload workinghitpayload;

		//if root node
		if (nodeIdxStack[stackPtr] == root_node_idx) {
			workinghitpayload = intersectAABB(ray, stackTopNode->m_BoundingBox);

			//miss
			if (workinghitpayload.hit_distance < 0) {
				continue;
			}

			nodeHitDist = workinghitpayload.hit_distance;
		}

		if (closest_hitpayload->primitiveptr != nullptr && closest_hitpayload->hit_distance < nodeHitDist) {
			continue;
		}

		closest_hitpayload->color += make_float3(1) * 0.05f;

		if (stackTopNode->m_IsLeaf) {
			for (int primIdx = 0; primIdx < stackTopNode->primitives_count; primIdx++) {
				const Triangle* primitive = stackTopNode->dev_primitive_ptrs_buffer[primIdx];
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
			HitPayload hit1 = intersectAABB(ray, (scenedata->DeviceBVHNodesBuffer[stackTopNode->dev_child1_idx]).m_BoundingBox);
			HitPayload hit2 = intersectAABB(ray, (scenedata->DeviceBVHNodesBuffer[stackTopNode->dev_child2_idx]).m_BoundingBox);

			if (hit1.hit_distance > hit2.hit_distance) {
				if (hit1.hit_distance >= 0) { nodeHitDistStack[stackPtr] = hit1.hit_distance; nodeIdxStack[stackPtr++] = stackTopNode->dev_child1_idx; }
				if (hit2.hit_distance >= 0) { nodeHitDistStack[stackPtr] = hit2.hit_distance; nodeIdxStack[stackPtr++] = stackTopNode->dev_child2_idx; }
			}
			else {
				if (hit2.hit_distance >= 0) { nodeHitDistStack[stackPtr] = hit2.hit_distance; nodeIdxStack[stackPtr++] = stackTopNode->dev_child2_idx; }
				if (hit1.hit_distance >= 0) { nodeHitDistStack[stackPtr] = hit1.hit_distance; nodeIdxStack[stackPtr++] = stackTopNode->dev_child1_idx; }
			}
		}
	}
}

//apparently do not bother sorting nodes for shadow rays and do an early out
__device__ bool traverseBVH_raytest(const Ray& ray, const int root_node_idx, const SceneData* scenedata) {
	if (root_node_idx < 0) return false;//empty scene

	const uint8_t maxStackSize = 64; // Adjust based on expected max depth
	int nodeIdxStack[maxStackSize];
	float nodeHitDistStack[maxStackSize];
	uint8_t stackPtr = 0;

	float nodeHitDist = FLT_MAX;//redundant?
	const BVHNode* stackTopNode = nullptr;

	nodeHitDistStack[stackPtr] = FLT_MAX;
	nodeIdxStack[stackPtr++] = root_node_idx;

	while (stackPtr > 0) {
		stackTopNode = &(scenedata->DeviceBVHNodesBuffer[nodeIdxStack[--stackPtr]]);
		nodeHitDist = nodeHitDistStack[stackPtr];

		HitPayload workinghitpayload;

		//if root node
		if (nodeIdxStack[stackPtr] == root_node_idx) {
			workinghitpayload = intersectAABB(ray, stackTopNode->m_BoundingBox);

			if (workinghitpayload.hit_distance < 0) {
				continue;
			}

			nodeHitDist = workinghitpayload.hit_distance;
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
			HitPayload hit1 = intersectAABB(ray, (scenedata->DeviceBVHNodesBuffer[stackTopNode->dev_child1_idx]).m_BoundingBox);
			HitPayload hit2 = intersectAABB(ray, (scenedata->DeviceBVHNodesBuffer[stackTopNode->dev_child2_idx]).m_BoundingBox);

			if (hit1.hit_distance > hit2.hit_distance) {
				if (hit1.hit_distance >= 0) { nodeHitDistStack[stackPtr] = hit1.hit_distance; nodeIdxStack[stackPtr++] = stackTopNode->dev_child1_idx; }
				if (hit2.hit_distance >= 0) { nodeHitDistStack[stackPtr] = hit2.hit_distance; nodeIdxStack[stackPtr++] = stackTopNode->dev_child2_idx; }
			}
			else {
				if (hit2.hit_distance >= 0) { nodeHitDistStack[stackPtr] = hit2.hit_distance; nodeIdxStack[stackPtr++] = stackTopNode->dev_child2_idx; }
				if (hit1.hit_distance >= 0) { nodeHitDistStack[stackPtr] = hit1.hit_distance; nodeIdxStack[stackPtr++] = stackTopNode->dev_child1_idx; }
			}
		}
	}

	return false;
}