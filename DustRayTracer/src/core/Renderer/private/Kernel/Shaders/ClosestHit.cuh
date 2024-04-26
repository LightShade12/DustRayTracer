#pragma once
#include "core/Renderer/private/CudaMath/helper_math.cuh"

__device__ HitPayload ClosestHit(const Ray& ray, uint32_t obj_idx, float hit_distance,
	const Mesh* meshBuffer, int triangleIdx) {
	Triangle triangle = meshBuffer[obj_idx].m_dev_triangles[triangleIdx];

	HitPayload payload;

	float3 hitpoint = ray.origin + ray.direction * hit_distance;

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

	payload.hit_distance = hit_distance;
	payload.world_position = ray.origin + ray.direction * hit_distance;//hit position
	payload.object_idx = obj_idx;

	if (dot(triangle.face_normal, ray.direction) > 0)
		payload.world_normal = -float3(triangle.face_normal);
	else
		payload.world_normal = triangle.face_normal;

	return payload;
};