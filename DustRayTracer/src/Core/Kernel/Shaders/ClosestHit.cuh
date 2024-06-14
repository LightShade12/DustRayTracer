#pragma once
#include "Core/CudaMath/helper_math.cuh"

__device__ HitPayload ClosestHit(const Ray& ray, const HitPayload* in_payload) {
	const Triangle* triangle = in_payload->primitiveptr;

	HitPayload payload;

	/*float3 hitpoint = ray.getOrigin() + ray.getDirection() * hit_distance;

	float3 v0v1 = triangle->vertex1.position - triangle->vertex0.position;
	float3 v0v2 = triangle->vertex2.position - triangle->vertex0.position;
	float3 v0p = hitpoint - triangle->vertex0.position;

	float d00 = dot(v0v1, v0v1);
	float d01 = dot(v0v1, v0v2);
	float d11 = dot(v0v2, v0v2);
	float d20 = dot(v0p, v0v1);
	float d21 = dot(v0p, v0v2);

	float denom = d00 * d11 - d01 * d01;
	payload.UVW.y = (d11 * d20 - d01 * d21) / denom;
	payload.UVW.z = (d00 * d21 - d01 * d20) / denom;
	payload.UVW.x = 1.0f - payload.UVW.y - payload.UVW.z;*/
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