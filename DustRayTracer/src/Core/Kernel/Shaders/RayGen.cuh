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

__device__ static float3 toneMapping(float3 HDR_color, float exposure = 2.f) {
	float3 LDR_color = uncharted2_filmic(HDR_color, exposure);
	return LDR_color;
}

__device__ static float3 gammaCorrection(const float3 linear_color) {
	float3 gamma_space_color = { sqrtf(linear_color.x),sqrtf(linear_color.y) ,sqrtf(linear_color.z) };
	return gamma_space_color;
}

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

//v is viewdir
__device__ float3 shlick_approx_fresnel_refl(float cos_theta, float3 f0) {
	//Shlick approx
	return f0 + (1 - f0) * powf(1.0f - cos_theta, 5);
}

//ggx ndf
__device__ float ggx_D(float NoH, float roughness) {
	float a = roughness * roughness;
	float a2 = a * a;
	float NoH2 = NoH * NoH;
	float b = (NoH2 * (a2 - 1) + 1);
	return a2 / (PI * b * b);
	//return a * a / (PI * powf(powf(NoH, 2) * ((a * a) - 1) + 1, 2));
}

__device__ float shlick_ggx(float NoV, float roughness) {
	//float nvd = fmaxf(0.001, AoB);
	float k = roughness * roughness / 2;
	return fmaxf(0.001, NoV) / (NoV * (1.f - k) + k);
}

//G factor from 0 to 1
	//slick ggx
__device__ float smith_shlick_ggx_G(float NoV, float NoL, float roughness) {
	return shlick_ggx(NoL, roughness) * shlick_ggx(NoV, roughness);
}

__device__ float3 BRDF(float3 incoming_lightdir, float3 outgoing_viewdir, float3 normal,
	const SceneData& scenedata, const Material& material, const float2& texture_uv) {
	float3 h = normalize(incoming_lightdir + outgoing_viewdir);

	float NoV = fminf(1, fmaxf(0, dot(normal, outgoing_viewdir)));
	float NoH = fminf(1, fmaxf(0, dot(normal, h)));
	float NoL = fminf(1, fmaxf(0, dot(normal, incoming_lightdir)));
	float VoH = fminf(1, fmaxf(0, dot(outgoing_viewdir, h)));

	float reflectance = scenedata.RenderSettings.global_reflectance;
	float metallic = scenedata.RenderSettings.global_metallic;
	float roughness = scenedata.RenderSettings.global_roughness;

	float3 basecolor;

	if (scenedata.RenderSettings.use_global_basecolor) basecolor = scenedata.RenderSettings.global_albedo;
	//else basecolor = material.Albedo;
	else
	{
		if (material.AlbedoTextureIndex < 0)basecolor = material.Albedo;
		else basecolor = scenedata.DeviceTextureBufferPtr[material.AlbedoTextureIndex].getPixel(texture_uv);
	}

	float3 f0 = make_float3(0.16f * (reflectance * reflectance));
	f0 = lerp(f0, basecolor, metallic);

	float3 F = shlick_approx_fresnel_refl(VoH, f0);
	float D = ggx_D(NoH, roughness);
	float G = smith_shlick_ggx_G(NoV, NoL, roughness);

	float3 spec = (F * D * G) / (4.0f * fmaxf(0.001, NoV) * fmaxf(0.001, NoL));
	float3 rhod = basecolor;
	rhod *= 1.f - F;//optional
	rhod *= (1.f - metallic);
	float3 diffuse = rhod / PI;

	return diffuse + spec;
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
	float3 previous_worldnormal{};
	float3 previous_ray_dir{};
	float3 next_ray_dir{};

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
				float3 sky_incoming_emission = skyModel(ray, scenedata) * scenedata.RenderSettings.sky_intensity;
				if (bounce_depth > 0)
					outgoing_light += sky_incoming_emission * cumulative_incoming_light_throughput;
				//* cosine_falloff_factor(normalize(ray.getDirection()), previous_worldnormal);
				else
					outgoing_light += sky_incoming_emission * cumulative_incoming_light_throughput;
			}
			break;
		}

		//SURFACE SHADING-------------------------------------------------------------------
		next_ray_dir = normalize(payload.world_normal + (randomUnitSphereVec3(seed)));
		current_material = &(scenedata.DeviceMaterialBufferPtr[payload.primitiveptr->material_idx]);
		if (current_material->AlbedoTextureIndex < 0) {
			if (bounce_depth > 0)
				cumulative_incoming_light_throughput *= BRDF(next_ray_dir, normalize(ray.getDirection()), payload.world_normal,
					scenedata, *current_material, texture_sample_uv)
				* cosine_falloff_factor(next_ray_dir, payload.world_normal);
			else
				cumulative_incoming_light_throughput *= BRDF(next_ray_dir, normalize(ray.getDirection()), payload.world_normal,
					scenedata, *current_material, texture_sample_uv);
		}
		else
		{
			const Triangle* tri = payload.primitiveptr;
			texture_sample_uv = payload.UVW.x * tri->vertex0.UV + payload.UVW.y * tri->vertex1.UV + payload.UVW.z * tri->vertex2.UV;
			if (bounce_depth > 0)
				cumulative_incoming_light_throughput *= current_material->EmmisiveFactor + BRDF(next_ray_dir, normalize(ray.getDirection()), payload.world_normal,
					scenedata, *current_material, texture_sample_uv)
				* cosine_falloff_factor(next_ray_dir, payload.world_normal);
			else
				cumulative_incoming_light_throughput *= current_material->EmmisiveFactor + BRDF(next_ray_dir, normalize(ray.getDirection()), payload.world_normal,
					scenedata, *current_material, texture_sample_uv);
		}

		//SHADOWRAY-------------------------------------------------------------------------------------------
		float3 new_ray_origin = payload.world_position + (payload.world_normal * 0.001f);

		//shadow ray for sunlight
		if (scenedata.RenderSettings.enableSunlight && scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::NORMALMODE)
		{
			if (!rayTest(Ray((new_ray_origin), (sunpos)+randomUnitVec3(seed) * 1.5),
				&scenedata))
				outgoing_light += suncol * cumulative_incoming_light_throughput *
				cosine_falloff_factor(normalize(sunpos), payload.world_normal);
		}

		//BOUNCE RAY---------------------------------------------------------------------------------------

		previous_worldnormal = payload.world_normal;
		previous_ray_dir = ray.getDirection();
		//diffuse scattering
		ray.setOrig(new_ray_origin);
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
	//--------------------------------------------------------------------------------

	//post processing
	if (scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::NORMALMODE || scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::ALBEDO_DEBUG)
	{
		if (scenedata.RenderSettings.tone_mapping)outgoing_light = toneMapping(outgoing_light, cam->exposure);
		if (scenedata.RenderSettings.gamma_correction)outgoing_light = gammaCorrection(outgoing_light);
	}

	return outgoing_light;
};