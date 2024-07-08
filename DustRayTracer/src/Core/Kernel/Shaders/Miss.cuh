#pragma once
__device__ HitPayload Miss(const Ray& ray, float3 bvh_debug_color) {
	HitPayload out_payload;
	out_payload.color = bvh_debug_color;
	return out_payload;
};