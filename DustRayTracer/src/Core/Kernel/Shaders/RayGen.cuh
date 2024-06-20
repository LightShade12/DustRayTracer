#pragma once
#include "Core/Kernel/TraceRay.cuh"

#include "Common/physical_units.hpp"
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

__device__ static float3 skyModel(const Ray& ray, const SceneData& scenedata) {
	float vertical_gradient_factor = 0.5 * (1 + (normalize(ray.getDirection())).y);//clamps to range 0-1
	float3 col1 = scenedata.RenderSettings.sky_color;
	float3 col2 = { 1,1,1 };
	float3 fcol = (float(1 - vertical_gradient_factor) * col2) + (vertical_gradient_factor * col1);
	fcol = { std::powf(fcol.x,2), std::powf(fcol.y,2) , std::powf(fcol.z,2) };
	return fcol;
}
//make sure the directions are facing the same general direction
__device__ inline float cosine_falloff_factor(float3 incoming_lightdir, float3 normal) {
	return fmaxf(0, dot(incoming_lightdir, normal));
}

__device__ float3 fresnelSchlick(float cosTheta, float3 F0) {
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

__device__ float D_GGX(float NoH, float roughness) {
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float NoH2 = NoH * NoH;
	float b = (NoH2 * (alpha2 - 1.0) + 1.0);
	return alpha2 * (1 / PI) / (b * b);
}

__device__ float G1_GGX_Schlick(float NoV, float roughness) {
	float alpha = roughness * roughness;
	float k = alpha / 2.0;
	return max(NoV, 0.001) / (NoV * (1.0 - k) + k);
}

__device__ float G_Smith(float NoV, float NoL, float roughness) {
	return G1_GGX_Schlick(NoL, roughness) * G1_GGX_Schlick(NoV, roughness);
}

__device__ float fresnelSchlick90(float cosTheta, float F0, float F90) {
	return F0 + (F90 - F0) * pow(1.0 - cosTheta, 5.0);
}

__device__ float disneyDiffuseFactor(float NoV, float NoL, float VoH, float roughness) {
	float alpha = roughness * roughness;
	float F90 = 0.5 + 2.0 * alpha * VoH * VoH;
	float F_in = fresnelSchlick90(NoL, 1.0, F90);
	float F_out = fresnelSchlick90(NoV, 1.0, F90);
	return F_in * F_out;
}

__device__ float3 BRDF(float3 incoming_lightdir, float3 outgoing_viewdir, float3 normal, const SceneData& scene_data,
	const Material& material, const float2& texture_uv) {
	float3 H = normalize(outgoing_viewdir + incoming_lightdir);

	float NoV = clamp(dot(normal, outgoing_viewdir), 0.0, 1.0);
	float NoL = clamp(dot(normal, incoming_lightdir), 0.0, 1.0);
	float NoH = clamp(dot(normal, H), 0.0, 1.0);
	float VoH = clamp(dot(outgoing_viewdir, H), 0.0, 1.0);

	float reflectance = scene_data.RenderSettings.global_reflectance;
	float roughness = scene_data.RenderSettings.global_roughness;
	float metallic = scene_data.RenderSettings.global_metallic;
	float3 baseColor;

	if (material.AlbedoTextureIndex < 0)baseColor = material.Albedo;
	else baseColor = scene_data.DeviceTextureBufferPtr[material.AlbedoTextureIndex].getPixel(texture_uv);

	float3 f0 = make_float3(0.16 * (reflectance * reflectance));
	f0 = lerp(f0, baseColor, metallic);

	float3 F = fresnelSchlick(VoH, f0);
	float D = D_GGX(NoH, roughness);
	float G = G_Smith(NoV, NoL, roughness);

	float3 spec = (F * D * G) / (4.0 * max(NoV, 0.001) * max(NoL, 0.001));

	float3 rhoD = baseColor;

	// optionally
	rhoD *= 1.0 - F;
	rhoD *= disneyDiffuseFactor(NoV, NoL, VoH, roughness);

	rhoD *= (1.0 - metallic);

	float3 diff = rhoD / PI;

	return diff + spec;
}

__device__ float3 rayGen(uint32_t x, uint32_t y, uint32_t max_x, uint32_t max_y,
	const Camera* cam, uint32_t frameidx, const SceneData scenedata) {
	float3 sunpos = make_float3(
		sin(scenedata.RenderSettings.sunlight_dir.x) * (1 - sin(scenedata.RenderSettings.sunlight_dir.y)),
		sin(scenedata.RenderSettings.sunlight_dir.y),
		cos(scenedata.RenderSettings.sunlight_dir.x) * (1 - sin(scenedata.RenderSettings.sunlight_dir.y))) * 100;
	float3 suncol = scenedata.RenderSettings.sunlight_color * scenedata.RenderSettings.sunlight_intensity;

	uint32_t seed = x + y * max_x;
	seed *= frameidx;

	float2 screen_uv = { (float(x) / max_x) ,(float(y) / max_y) };
	screen_uv = screen_uv * 2 - 1;
	Ray ray = cam->getRay(screen_uv, max_x, max_y, seed);
	ray.interval = Interval(-1, FLT_MAX);

	float3 outgoing_light = { 0,0,0 };
	float3 cumulative_incoming_light_throughput = { 1,1,1 };
	int bounces = scenedata.RenderSettings.ray_bounce_limit;

	HitPayload payload;
	float2 texture_sample_uv = { 0,1 };//DEBUG
	const Material* current_material = nullptr;

	for (int bounce_depth = 0; bounce_depth <= bounces; bounce_depth++)
	{
		payload = traceRay(ray, &scenedata);
		seed += bounce_depth;

		//SHADING------------------------------------------------------------
		//SKY SHADING------------------------
		if (payload.primitiveptr == nullptr)
		{
			if (scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::MESHBVH_DEBUG &&
				scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::DEBUGMODE)outgoing_light = payload.color;
			else {
				float3 sky_integral_eval = skyModel(ray, scenedata) * scenedata.RenderSettings.sky_intensity;
				outgoing_light += sky_integral_eval * cumulative_incoming_light_throughput;
			}
			break;
		}

		//SURFACE SHADING-------------------------------------------------------------------
		float3 next_ray_origin = payload.world_position + (payload.world_normal * 0.001f);
		float3 next_ray_dir = normalize(payload.world_normal + randomUnitSphereVec3(seed));

		current_material = &(scenedata.DeviceMaterialBufferPtr[payload.primitiveptr->material_idx]);

		const Triangle* tri = payload.primitiveptr;
		texture_sample_uv = payload.UVW.x * tri->vertex0.UV + payload.UVW.y * tri->vertex1.UV + payload.UVW.z * tri->vertex2.UV;
		float3 lightdir = next_ray_dir;
		float3 viewdir = -1.f * (ray.getDirection());
		cumulative_incoming_light_throughput *= (current_material->EmmisiveFactor * 10) +
			BRDF(lightdir, viewdir, payload.world_normal,
				scenedata, *current_material, texture_sample_uv)
			* cosine_falloff_factor(lightdir, payload.world_normal);

		//SHADOWRAY-------------------------------------------------------------------------------------------

		//shadow ray for sunlight
		if (scenedata.RenderSettings.enableSunlight && scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::NORMALMODE)
		{
			if (!rayTest(Ray((next_ray_origin), (sunpos)+randomUnitFloat3(seed) * 1.5),
				&scenedata))
				outgoing_light += suncol * cumulative_incoming_light_throughput *
				cosine_falloff_factor(normalize(sunpos), payload.world_normal);
			//cumulative_incoming_light_throughput *= suncol * cosine_falloff_factor(normalize(sunpos), payload.world_normal);
		}

		//BOUNCE RAY---------------------------------------------------------------------------------------

		ray.setOrig(next_ray_origin);
		ray.setDir(next_ray_dir);

		//Debug Views------------------------------------------------------------------------------------
		if (scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::DEBUGMODE)
		{
			switch (scenedata.RenderSettings.DebugMode)
			{
			case RendererSettings::DebugModes::ALBEDO_DEBUG:
				outgoing_light = cumulative_incoming_light_throughput; break;

			case RendererSettings::DebugModes::NORMAL_DEBUG:
				outgoing_light = payload.world_normal; break;

			case RendererSettings::DebugModes::BARYCENTRIC_DEBUG:
				outgoing_light = payload.UVW; break;

			case RendererSettings::DebugModes::UVS_DEBUG:
				outgoing_light = { texture_sample_uv.x,texture_sample_uv.y,0 }; break;

			case RendererSettings::DebugModes::MESHBVH_DEBUG:
				outgoing_light = { 0,0.1,0.1 };
				outgoing_light += payload.color; break;

			default:
				break;
			}
			break;
		}
	}

	return outgoing_light;
};