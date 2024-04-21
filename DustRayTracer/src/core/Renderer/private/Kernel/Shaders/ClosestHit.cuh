#pragma once
#include "core/Renderer/private/CudaMath/helper_math.cuh"

__device__ HitPayload ClosestHit(const Ray& ray, uint32_t obj_idx, float hit_distance, const Triangle* scene_vec) {
	const Triangle triangle = (scene_vec[obj_idx]);

	HitPayload payload;
	float3 edge1, edge2;

	edge1 = triangle.vertex1 - triangle.vertex0;

	edge2 = triangle.vertex2 - triangle.vertex0;

	payload.hit_distance = hit_distance;
	payload.world_position = ray.origin + ray.direction * hit_distance;//hit position
	payload.object_idx = obj_idx;

	payload.world_normal = -normalize(cross(edge1, edge2));

	return payload;
};