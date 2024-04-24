#pragma once
#include "core/Renderer/private/CudaMath/helper_math.cuh"

__device__ HitPayload ClosestHit(const Ray& ray, uint32_t obj_idx, float hit_distance,
	const Mesh* meshBuffer, int triangleIdx) {
	Triangle triangle = meshBuffer[obj_idx].m_dev_triangles[triangleIdx];

	HitPayload payload;

	payload.hit_distance = hit_distance;
	payload.world_position = ray.origin + ray.direction * hit_distance;//hit position
	payload.object_idx = obj_idx;

	if (dot(triangle.normal, ray.direction) > 0)
		payload.world_normal = -float3(triangle.normal);
	else
		payload.world_normal = triangle.normal;

	return payload;
};