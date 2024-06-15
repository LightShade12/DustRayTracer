#pragma once
#include "Core/CudaMath/helper_math.cuh"

__device__ HitPayload ClosestHit(const Ray& ray, const HitPayload* in_payload) {
	const Triangle* triangle = in_payload->primitiveptr;

	HitPayload payload;

	payload.UVW = in_payload->UVW;
	//payload.triangle_idx = triangleIdx;
	payload.primitiveptr = in_payload->primitiveptr;
	payload.hit_distance = in_payload->hit_distance;
	payload.world_position = ray.getOrigin() + ray.getDirection() * in_payload->hit_distance;
	payload.color = in_payload->color;
	//payload.object_idx = obj_idx;

	if (dot(triangle->face_normal, normalize(ray.getDirection())) > 0.f)
	{
		payload.front_face = false;
		payload.world_normal = -1.f * triangle->face_normal;
	}
	else {
		payload.world_normal = triangle->face_normal;
		payload.front_face = true;
	}

	return payload;
};