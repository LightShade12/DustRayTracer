#pragma once
__device__ HitPayload Miss(const Ray& ray, float3 bvh_debug_color) {
	HitPayload payload;
	payload.color = bvh_debug_color;
	payload.hit_distance = -1;
	return payload;
};