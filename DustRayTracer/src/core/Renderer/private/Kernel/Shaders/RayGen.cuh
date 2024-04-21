#pragma once
#include "core/Renderer/private/Kernel/TraceRay.cuh"

#include <core/Renderer/private/CudaMath/helper_math.cuh>
#include <core/Renderer/private/CudaMath/Random.cuh>

#include <core/Renderer/private/Kernel/Ray.cuh>
#include <core/Renderer/private/Kernel/HitPayload.cuh>
#include <core/Renderer/private/Shapes/Scene.cuh>


__device__ float3 RayGen(uint32_t x, uint32_t y, uint32_t max_x, uint32_t max_y,
	const Camera* cam, const Triangle* scene_vector, size_t scenevecsize, const Material* matvector, uint32_t frameidx) {
	float2 uv = { (float(x) / max_x) ,(float(y) / max_y) };

	float3 sunpos = { 100,100,100 };
	float3 suncol = { 1.0, 1.0, 0.584 };
	//uv.x *= ((float)max_x / (float)max_y);
	//uv.x = uv.x * 2.f - ((float)max_x / (float)max_y);
	//uv.y = uv.y * 2.f - 1.f;
	uv = uv * 2 - 1;

	Ray ray;
	ray.origin = cam->m_Position;
	ray.direction = cam->GetRayDir(uv, 30, max_x, max_y);
	float3 light = { 0,0,0 };

	uint32_t seed = x + y * max_x;
	seed *= frameidx;

	float3 contribution = { 1,1,1 };
	int bounces = 1;

	for (int i = 0; i < bounces; i++)
	{
		HitPayload payload = TraceRay(ray, scene_vector, scenevecsize);
		seed += i;
		//sky
		if (payload.hit_distance < 0)
		{
			float a = 0.5 * (1 + (normalize(ray.direction)).y);
			float3 col1 = { 0.2,0.5,1.0 };
			float3 col2 = { 1,1,1 };
			float3 fcol = (float(1 - a) * col2) + (a * col1);
			light += fcol * contribution;
			break;
		}

		float lightIntensity = max(dot(payload.world_normal, sunpos), 0.0f); // == cos(angle)

		const Triangle closestTriangle = scene_vector[payload.object_idx];
		const Material material = matvector[closestTriangle.MaterialIdx];
		//light = material.Albedo;
		contribution *= material.Albedo;

		float3 newRayOrigin = payload.world_position + (payload.world_normal * 0.0001f);
		HitPayload shadowpayload = TraceRay(Ray(newRayOrigin, (sunpos - newRayOrigin) + randomUnitVec3(seed) * 2), scene_vector, scenevecsize);
		if (shadowpayload.hit_distance < 0)
		{
			light += (material.Albedo * suncol) * 0.5;
		}

		ray.origin = newRayOrigin;
		ray.direction = payload.world_normal + (normalize(randomUnitSphereVec3(seed)));

		light = { payload.world_normal.x, payload.world_normal.y, payload.world_normal.z };//debug normals
	}

	//light = { sqrtf(light.x),sqrtf(light.y) ,sqrtf(light.z) };//uses 1/gamma=2 not 2.2
	light = fminf(light, { 1,1,1 });
	return light;
};