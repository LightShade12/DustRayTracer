#pragma once
#include "Core/Kernel/TraceRay.cuh"

#include "Core/CudaMath/helper_math.cuh"
#include "Core/CudaMath/Random.cuh"

#include "Core/Ray.cuh"
#include "Core/HitPayload.cuh"
#include "Core/Scene/Scene.cuh"

__device__ float3 RayGen(uint32_t x, uint32_t y, uint32_t max_x, uint32_t max_y,
	const Camera* cam, uint32_t frameidx, const SceneData scenedata) {
	float2 uv = { (float(x) / max_x) ,(float(y) / max_y) };

	float3 sunpos = make_float3(
		sin(scenedata.RenderSettings.sunlight_dir.x) * (1 - sin(scenedata.RenderSettings.sunlight_dir.y)),
		sin(scenedata.RenderSettings.sunlight_dir.y),
		cos(scenedata.RenderSettings.sunlight_dir.x) * (1 - sin(scenedata.RenderSettings.sunlight_dir.y))) * 100;
	float3 suncol = scenedata.RenderSettings.sunlight_color * scenedata.RenderSettings.sunlight_intensity;
	//uv.x *= ((float)max_x / (float)max_y);
	//uv.x = uv.x * 2.f - ((float)max_x / (float)max_y);
	//uv.y = uv.y * 2.f - 1.f;
	uv = uv * 2 - 1;

	Ray ray;
	ray.origin = cam->m_Position;
	ray.direction = cam->GetRayDir(uv, max_x, max_y);
	float3 light = { 0,0,0 };

	uint32_t seed = x + y * max_x;
	seed *= frameidx;

	float3 contribution = { 1,1,1 };
	int bounces = scenedata.RenderSettings.ray_bounce_limit;

	for (int i = 0; i <= bounces; i++)
	{
		float2 uv = { 0,1 };//DEBUG
		HitPayload payload = TraceRay(ray, &scenedata);
		seed += i;

		if (payload.debug)
		{
			return make_float3(1, 0, 1);
		}

		//sky
		if (payload.hit_distance < 0)
		{
			float a = 0.5 * (1 + (normalize(ray.direction)).y);
			float3 col1 = scenedata.RenderSettings.sky_color * scenedata.RenderSettings.sky_intensity;
			float3 col2 = { 1,1,1 };
			float3 fcol = (float(1 - a) * col2) + (a * col1);
			light += fcol * contribution;
			if (scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::MESHBVH_DEBUG && scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::DEBUGMODE)
			{
				light = { 0,0,0 };
				light += payload.color;
			}
			break;
		}

		float lightIntensity = max(dot(payload.world_normal, sunpos), 0.0f); // == cos(angle)

		Material material = scenedata.DeviceMaterialBufferPtr[payload.primitiveptr->materialIdx];

		//printf("kernel texture idx eval: %d ", material.AlbedoTextureIndex);

		if (material.AlbedoTextureIndex < 0)
		{
			contribution *= material.Albedo;
		}
		else
		{
			//Triangle tri = closestMesh.m_dev_triangles[payload.triangle_idx];
			const Triangle tri = *(payload.primitiveptr);
			uv = {
				 payload.UVW.x * tri.vertex0.UV.x + payload.UVW.y * tri.vertex1.UV.x + payload.UVW.z * tri.vertex2.UV.x,
				  payload.UVW.x * tri.vertex0.UV.y + payload.UVW.y * tri.vertex1.UV.y + payload.UVW.z * tri.vertex2.UV.y
			};
			contribution *= scenedata.DeviceTextureBufferPtr[material.AlbedoTextureIndex].getPixel(uv);
		}

		float3 newRayOrigin = payload.world_position + (payload.world_normal * 0.0001f);

		//shadow ray for sunlight
		if (scenedata.RenderSettings.enableSunlight && scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::NORMALMODE)
			if (i < 2)
				if (!RayTest(Ray(newRayOrigin, (sunpos - newRayOrigin) + randomUnitVec3(seed) * 2),
					&scenedata))
				{
					if (material.AlbedoTextureIndex < 0)
						light += (material.Albedo * suncol);
					else
					{
						light += scenedata.DeviceTextureBufferPtr[material.AlbedoTextureIndex].getPixel(uv) * suncol;
					}
				}

		ray.origin = newRayOrigin;
		ray.direction = payload.world_normal + (normalize(randomUnitSphereVec3(seed)));

		if (scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::DEBUGMODE)
		{
			if (scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::ALBEDO_DEBUG)
			{
				if (material.AlbedoTextureIndex < 0)
				{
					light = material.Albedo;
				}
				else
				{
					Triangle tri = *(payload.primitiveptr);
					uv = {
						 payload.UVW.x * tri.vertex0.UV.x + payload.UVW.y * tri.vertex1.UV.x + payload.UVW.z * tri.vertex2.UV.x,
						  payload.UVW.x * tri.vertex0.UV.y + payload.UVW.y * tri.vertex1.UV.y + payload.UVW.z * tri.vertex2.UV.y
					};
					light = scenedata.DeviceTextureBufferPtr[material.AlbedoTextureIndex].getPixel(uv);
				}
			}
			if (scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::NORMAL_DEBUG)
				light = payload.world_normal;//debug normals
			if (scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::BARYCENTRIC_DEBUG)
				light = payload.UVW;//debug barycentric coords
			if (scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::UVS_DEBUG)
				light = { uv.x,uv.y,0 };//debug UV
			if (scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::MESHBVH_DEBUG)
			{
				light = { 0,0.1,0.1 };
				light += payload.color;
			}
		}
		if (scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::DEBUGMODE)break;
	}

	if (scenedata.RenderSettings.gamma_correction &&
		(scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::NORMALMODE || scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::ALBEDO_DEBUG))
		light = { sqrtf(light.x),sqrtf(light.y) ,sqrtf(light.z) };//uses 1/gamma=2 not 2.2

	return light;
};