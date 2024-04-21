#pragma once
__device__ HitPayload Miss(const Ray& ray) {
	HitPayload payload;
	payload.hit_distance = -1;
	return payload;
};
