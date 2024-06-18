#pragma once
#include "Core/Kernel/TraceRay.cuh"

#include "Core/CudaMath/helper_math.cuh"
#include "Core/CudaMath/Random.cuh"

#include "Core/Ray.cuh"
#include "Core/HitPayload.cuh"
#include "Core/Scene/Scene.cuh"
#include "Core/Scene/Camera.cuh"

/*
TODO: List of things:
-DLSS 3.5 like features
-make DRT a separate project
-reuse pipelin data
-make floating vars into class
-cleanup camera code
-add pbrt objects
-reuse a lot of vars in payload; like world_position
*/

__device__ static float3 uncharted2_tonemap_partial(float3 x)
{
	float A = 0.15f;
	float B = 0.50f;
	float C = 0.10f;
	float D = 0.20f;
	float E = 0.02f;
	float F = 0.30f;
	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

__device__ static float3 uncharted2_filmic(float3 v, float exposure)
{
	float exposure_bias = exposure;
	float3 curr = uncharted2_tonemap_partial(v * exposure_bias);

	float3 W = make_float3(11.2f);
	float3 white_scale = make_float3(1.0f) / uncharted2_tonemap_partial(W);
	return curr * white_scale;
}

__device__ static float3 tonemapper(float3 HDR_color, float exposure = 2.f) {
	float3 LDR_color = uncharted2_filmic(HDR_color, exposure);
	return LDR_color;
}

__device__ static float3 gammaCorrection(const float3 linear_color) {
	float3 gamma_space_color = { sqrtf(linear_color.x),sqrtf(linear_color.y) ,sqrtf(linear_color.z) };
	return gamma_space_color;
}

__device__ static float3 SkyModel(const Ray& ray, const SceneData& scenedata) {
	float vertical_gradient_factor = 0.5 * (1 + (normalize(ray.getDirection())).y);//clamps to range 0-1
	float3 col1 = scenedata.RenderSettings.sky_color;
	float3 col2 = { 1,1,1 };
	float3 fcol = (float(1 - vertical_gradient_factor) * col2) + (vertical_gradient_factor * col1);
	fcol = { std::powf(fcol.x,2), std::powf(fcol.y,2) , std::powf(fcol.z,2) };
	return fcol;
}

__device__ float3 RayGen(uint32_t x, uint32_t y, uint32_t max_x, uint32_t max_y,
	const Camera* cam, uint32_t frameidx, const SceneData scenedata) {
	float2 screen_uv = { (float(x) / max_x) ,(float(y) / max_y) };
	screen_uv = screen_uv * 2 - 1;

	float3 sunpos = make_float3(
		sin(scenedata.RenderSettings.sunlight_dir.x) * (1 - sin(scenedata.RenderSettings.sunlight_dir.y)),
		sin(scenedata.RenderSettings.sunlight_dir.y),
		cos(scenedata.RenderSettings.sunlight_dir.x) * (1 - sin(scenedata.RenderSettings.sunlight_dir.y))) * 100;
	float3 suncol = scenedata.RenderSettings.sunlight_color * scenedata.RenderSettings.sunlight_intensity;

	uint32_t seed = x + y * max_x;
	seed *= frameidx;

	Ray ray = cam->GetRay(screen_uv, max_x, max_y, seed);
	ray.interval = Interval(-1, FLT_MAX);

	float3 light = { 0,0,0 };
	float3 throughput = { 1,1,1 };
	int bounces = scenedata.RenderSettings.ray_bounce_limit;

	HitPayload payload;
	float2 texture_sample_uv = { 0,1 };//DEBUG
	const Material* current_material = nullptr;
	float3 lastworldnormal;

	for (int bounce_depth = 0; bounce_depth <= bounces; bounce_depth++)
	{
		payload = TraceRay(ray, &scenedata);
		seed += bounce_depth;

		//SHADING------------------------------------------------------------
		/*if (payload.debug)
		{
			return make_float3(1, 0, 1);
		}*/
		//SKY SHADING------------------------
		if (payload.primitiveptr == nullptr)
		{
			if (scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::MESHBVH_DEBUG &&
				scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::DEBUGMODE)light = payload.color;
			else {
				float3 skycolor = SkyModel(ray, scenedata);
				if (bounce_depth > 0) {
					light += skycolor * throughput * scenedata.RenderSettings.sky_intensity * fmaxf(0, dot(normalize(ray.getDirection()), lastworldnormal));
				}
				else
					light += skycolor * throughput * scenedata.RenderSettings.sky_intensity;
			}
			break;
		}

		//SURFACE SHADING-------------------------------------------------------------------
		current_material = &(scenedata.DeviceMaterialBufferPtr[payload.primitiveptr->materialIdx]);
		if (current_material->AlbedoTextureIndex < 0)throughput *= current_material->Albedo;
		else
		{
			const Triangle* tri = payload.primitiveptr;
			texture_sample_uv = payload.UVW.x * tri->vertex0.UV + payload.UVW.y * tri->vertex1.UV + payload.UVW.z * tri->vertex2.UV;
			if (bounce_depth > 0)
				throughput *= scenedata.DeviceTextureBufferPtr[current_material->AlbedoTextureIndex].getPixel(texture_sample_uv)
				* fmaxf(0, dot(normalize(ray.getDirection()), lastworldnormal));
			else
				throughput *= scenedata.DeviceTextureBufferPtr[current_material->AlbedoTextureIndex].getPixel(texture_sample_uv);
		}

		//SHADOWRAY-------------------------------------------------------------------------------------------
		float3 newRayOrigin = payload.world_position + (payload.world_normal * 0.001f);

		//shadow ray for sunlight
		if (scenedata.RenderSettings.enableSunlight && scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::NORMALMODE)
		{
			if (!RayTest(Ray((newRayOrigin), (sunpos)+randomUnitVec3(seed) * 1.5),
				&scenedata))
				light += suncol * throughput * fmaxf(0, dot(normalize(sunpos), payload.world_normal));
		}

		//BOUNCE RAY---------------------------------------------------------------------------------------

		lastworldnormal = payload.world_normal;
		//diffuse scattering
		ray.setOrig(newRayOrigin);
		ray.setDir(normalize(payload.world_normal + (randomUnitSphereVec3(seed))));

		//Debug Views------------------------------------------------------------------------------------
		if (scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::DEBUGMODE)
		{
			switch (scenedata.RenderSettings.DebugMode)
			{
			case RendererSettings::DebugModes::ALBEDO_DEBUG:
				light = throughput; break;

			case RendererSettings::DebugModes::NORMAL_DEBUG:
				light = payload.world_normal; break;

			case RendererSettings::DebugModes::BARYCENTRIC_DEBUG:
				light = payload.UVW; break;

			case RendererSettings::DebugModes::UVS_DEBUG:
				light = { texture_sample_uv.x,texture_sample_uv.y,0 }; break;

			case RendererSettings::DebugModes::MESHBVH_DEBUG:
				light = { 0,0.1,0.1 };
				light += payload.color; break;

			default:
				break;
			}
			break;
		}
	}

	//post processing
	if (scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::NORMALMODE || scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::ALBEDO_DEBUG)
	{
		if (scenedata.RenderSettings.tone_mapping)light = tonemapper(light, cam->exposure);
		if (scenedata.RenderSettings.gamma_correction)light = gammaCorrection(light);
	}

	return light;
};