#pragma once
#include "Core/CudaMath/helper_math.cuh"

__device__ HitPayload ClosestHit(const Ray& ray, float hit_distance, const Triangle* primitive, float3 bvh_debug_color) {
	const Triangle& triangle = *primitive;

	HitPayload payload;

	float3 hitpoint = ray.getOrigin() + ray.getDirection() * hit_distance;

	float3 v0v1 = triangle.vertex1.position - triangle.vertex0.position;
	float3 v0v2 = triangle.vertex2.position - triangle.vertex0.position;
	float3 v0p = hitpoint - triangle.vertex0.position;

	float d00 = dot(v0v1, v0v1);
	float d01 = dot(v0v1, v0v2);
	float d11 = dot(v0v2, v0v2);
	float d20 = dot(v0p, v0v1);
	float d21 = dot(v0p, v0v2);

	float denom = d00 * d11 - d01 * d01;
	payload.UVW.y = (d11 * d20 - d01 * d21) / denom;
	payload.UVW.z = (d00 * d21 - d01 * d20) / denom;
	payload.UVW.x = 1.0f - payload.UVW.y - payload.UVW.z;

	//payload.triangle_idx = triangleIdx;
	payload.primitiveptr = primitive;
	payload.hit_distance = hit_distance;
	payload.world_position = ray.getOrigin() + ray.getDirection() * hit_distance;
	//payload.object_idx = obj_idx;

	if (dot(triangle.face_normal, ray.getDirection()) > 0)
	{
		payload.front_face = false;
		payload.world_normal = -float3(triangle.face_normal);
	}
	else
		payload.world_normal = triangle.face_normal;

	payload.color = bvh_debug_color;
	return payload;
};