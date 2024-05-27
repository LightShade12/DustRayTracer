#pragma once
#include "BVHNode.cuh"
#include "core/Renderer/private/Kernel/Shaders/Intersection.cuh"
#include "core/Renderer/private/Kernel/Shaders/Anyhit.cuh"
#include "core/Renderer/private/CudaMath/helper_math.cuh"

__device__ HitPayload intersectAABB(const Ray& ray, const Bounds3f& bbox) {
	HitPayload payload;

	float3 invDir = 1.0f / ray.direction;
	float3 t0 = (bbox.pMin - ray.origin) * invDir;
	float3 t1 = (bbox.pMax - ray.origin) * invDir;

	float3 tmin = fminf(t0, t1);
	float3 tmax = fmaxf(t0, t1);

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
__device__ void traverseBVH(const Ray& ray, const BVHNode* root, HitPayload* closest_hitpayload, bool& debug, const SceneData* scenedata) {
	if (root == nullptr) return;

	// Explicit stack for iterative traversal
	const int maxStackSize = 64; // Adjust based on expected max depth
	const BVHNode* nodeStack[maxStackSize];
	int stackPtr = 0;

	nodeStack[stackPtr++] = root;

	while (stackPtr > 0) {
		const BVHNode* currentNode = nodeStack[--stackPtr];

		HitPayload workinghitpayload = intersectAABB(ray, currentNode->m_BoundingBox);
		if (workinghitpayload.hit_distance < 0) {
			continue;
		}

		if (closest_hitpayload->primitiveptr != nullptr && workinghitpayload.hit_distance > closest_hitpayload->hit_distance) {
			continue;
		}

		closest_hitpayload->color += make_float3(1)*0.05f;
		if (currentNode->m_IsLeaf) {
			for (int primIdx = 0; primIdx < currentNode->primitives_count; primIdx++) {
				const Triangle* prim = currentNode->dev_primitive_ptrs_buffer[primIdx];
				workinghitpayload = Intersection(ray, prim);

				if (workinghitpayload.primitiveptr != nullptr && workinghitpayload.hit_distance < closest_hitpayload->hit_distance) {
					if (!AnyHit(ray, scenedata, 
						prim, workinghitpayload.hit_distance))continue;
					closest_hitpayload->hit_distance = workinghitpayload.hit_distance;
					closest_hitpayload->primitiveptr = workinghitpayload.primitiveptr;
				}
			}
		}
		else {
			// Compute distances for child nodes
			HitPayload hit1 = intersectAABB(ray, currentNode->dev_child1->m_BoundingBox);
			HitPayload hit2 = intersectAABB(ray, currentNode->dev_child2->m_BoundingBox);

			// Push child nodes to the stack in order based on hit distance
			if (hit1.hit_distance > hit2.hit_distance) {
				if (hit1.hit_distance >= 0) nodeStack[stackPtr++] = currentNode->dev_child1;
				if (hit2.hit_distance >= 0) nodeStack[stackPtr++] = currentNode->dev_child2;
			}
			else {
				if (hit2.hit_distance >= 0) nodeStack[stackPtr++] = currentNode->dev_child2;
				if (hit1.hit_distance >= 0) nodeStack[stackPtr++] = currentNode->dev_child1;
			}
		}
	}
}

__device__ bool traverseBVH_raytest(const Ray& ray, const BVHNode* root, const SceneData* scenedata) {
	if (root == nullptr) return false;

	// Explicit stack for iterative traversal
	const int maxStackSize = 64; // Adjust based on expected max depth
	const BVHNode* nodeStack[maxStackSize];
	int stackPtr = 0;

	nodeStack[stackPtr++] = root;

	while (stackPtr > 0) {
		const BVHNode* currentNode = nodeStack[--stackPtr];

		HitPayload workinghitpayload = intersectAABB(ray, currentNode->m_BoundingBox);
		if (workinghitpayload.hit_distance < 0) {
			continue;
		}

		if (currentNode->m_IsLeaf) {
			for (int primIdx = 0; primIdx < currentNode->primitives_count; primIdx++) {
				const Triangle* prim = currentNode->dev_primitive_ptrs_buffer[primIdx];
				workinghitpayload = Intersection(ray, prim);

				if (workinghitpayload.primitiveptr != nullptr) {
					if (AnyHit(ray, scenedata, prim, workinghitpayload.hit_distance)) {
						return true; // Intersection found, return true immediately
					}
				}
			}
		}
		else {
			// Compute distances for child nodes
			HitPayload hit1 = intersectAABB(ray, currentNode->dev_child1->m_BoundingBox);
			HitPayload hit2 = intersectAABB(ray, currentNode->dev_child2->m_BoundingBox);

			// Push child nodes to the stack in order based on hit distance
			if (hit1.hit_distance > hit2.hit_distance) {
				if (hit1.hit_distance >= 0) nodeStack[stackPtr++] = currentNode->dev_child1;
				if (hit2.hit_distance >= 0) nodeStack[stackPtr++] = currentNode->dev_child2;
			}
			else {
				if (hit2.hit_distance >= 0) nodeStack[stackPtr++] = currentNode->dev_child2;
				if (hit1.hit_distance >= 0) nodeStack[stackPtr++] = currentNode->dev_child1;
			}
		}
	}

	// No intersection found, return false
	return false;
}