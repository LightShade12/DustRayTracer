#pragma once
#include "Core/CudaMath/helper_math.cuh"

__device__ HitPayload ClosestHit(const Ray& ray, const HitPayload* in_payload, const SceneData* scene_data) {
	const Triangle* triangle = &(scene_data->DevicePrimitivesBuffer[in_payload->triangle_idx]);

	HitPayload out_payload;

	out_payload.UVW = in_payload->UVW;
	//out_payload.triangle_idx = triangleIdx;
	out_payload.triangle_idx = in_payload->triangle_idx;
	out_payload.hit_distance = in_payload->hit_distance;
	out_payload.world_position = ray.getOrigin() + ray.getDirection() * in_payload->hit_distance;
	out_payload.color = in_payload->color;
	//out_payload.object_idx = obj_idx;

	if (dot(triangle->face_normal, normalize(ray.getDirection())) > 0.f)
	{
		out_payload.front_face = false;
		out_payload.world_normal = -1.f * triangle->face_normal;
	}
	else {
		out_payload.world_normal = triangle->face_normal;
		out_payload.front_face = true;
	}

	return out_payload;
};