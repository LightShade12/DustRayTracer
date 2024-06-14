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

__device__ static float3 luminanceCallibration() {
}

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

	float3 sunpos = make_float3(
		sin(scenedata.RenderSettings.sunlight_dir.x) * (1 - sin(scenedata.RenderSettings.sunlight_dir.y)),
		sin(scenedata.RenderSettings.sunlight_dir.y),
		cos(scenedata.RenderSettings.sunlight_dir.x) * (1 - sin(scenedata.RenderSettings.sunlight_dir.y))) * 100;
	float3 suncol = scenedata.RenderSettings.sunlight_color * scenedata.RenderSettings.sunlight_intensity;
	//uv.x *= ((float)max_x / (float)max_y);
	//uv.x = uv.x * 2.f - ((float)max_x / (float)max_y);
	//uv.y = uv.y * 2.f - 1.f;
	screen_uv = screen_uv * 2 - 1;

	uint32_t seed = x + y * max_x;
	seed *= frameidx;

	Ray ray = cam->GetRay(screen_uv, max_x, max_y, seed);
	ray.interval = Interval(-1, FLT_MAX);
	float3 light = { 0,0,0 };

	float3 throughput = { 1,1,1 };
	int bounces = scenedata.RenderSettings.ray_bounce_limit;
	float2 texture_sample_uv = { 0,1 };//DEBUG
	HitPayload payload;
	const Material* current_material = nullptr;

	for (int i = 0; i <= bounces; i++)
	{
		payload = TraceRay(ray, &scenedata);
		seed += i;

		//SHADING------------------------------------------------------------

		if (payload.debug)
		{
			return make_float3(1, 0, 1);
		}

		//SKY SHADING------------------------
		if (payload.primitiveptr == nullptr)
		{
			if (scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::MESHBVH_DEBUG &&
				scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::DEBUGMODE)
			{
				light = payload.color;
			}
			else
			{
				float3 skycolor = SkyModel(ray, scenedata);
				light += skycolor * throughput * scenedata.RenderSettings.sky_intensity;
			}
			break;
		}

		current_material = &(scenedata.DeviceMaterialBufferPtr[payload.primitiveptr->materialIdx]);

		//SURFACE SHADING-------------------------------------------------------------------
		if (current_material->AlbedoTextureIndex < 0)
		{
			if (current_material->Transmission)
				throughput *= make_float3(.9f);
			else
			{
				//TODO: Bug: Emissive material comtrib on diffuse isn't visisble when tonemapper is used
				throughput *= current_material->Albedo;
				if (!(current_material->EmmisiveFactor.x == 0 && current_material->EmmisiveFactor.y == 0 && current_material->EmmisiveFactor.z == 0)) {
					light += throughput * current_material->EmmisiveFactor * 1000;
					break;
				}
			}
		}
		else
		{
			if (current_material->Transmission)
				throughput *= make_float3(.9f);
			else
			{
				//Triangle tri = closestMesh.m_dev_triangles[payload.triangle_idx];
				const Triangle* tri = payload.primitiveptr;
				texture_sample_uv = payload.UVW.x * tri->vertex0.UV + payload.UVW.y * tri->vertex1.UV + payload.UVW.z * tri->vertex2.UV;
				throughput *= scenedata.DeviceTextureBufferPtr[current_material->AlbedoTextureIndex].getPixel(texture_sample_uv);
			}
		}

		//SHADOWRAY-------------------------------------------------------------------------------------------
		float3 newRayOrigin = payload.world_position + (payload.world_normal * 0.001f);

		//shadow ray for sunlight
		if (scenedata.RenderSettings.enableSunlight && scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::NORMALMODE)
		{
			if (!current_material->Metallic || !current_material->Transmission) {
				if (!RayTest(Ray((newRayOrigin), (sunpos)+randomUnitVec3(seed) * 1.5),
					&scenedata))
				{
					light += suncol * throughput;
				}
			}
			else {}
		}

		//BOUNCE RAY---------------------------------------------------------------------------------------

		ray.setOrig(newRayOrigin);
		if (current_material->Transmission) {
			//TODO: blue light specular should precede over the albedo at grazing angles
			//internal multi refraction shouldn't darken light; should stay constant throughout medium
			float ri = (payload.front_face) ? (1.f / current_material->refractive_index) : current_material->refractive_index;

			float3 unit_direction = normalize(ray.getDirection());
			float cos_theta = fmin(dot(-unit_direction, payload.world_normal), 1.0f);
			float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

			bool cannot_refract = ri * sin_theta > 1.0f;
			float3 direction;

			if (cannot_refract || reflectance(cos_theta, ri) > randomFloat(seed) * 0.35)
				direction = reflect(unit_direction, payload.world_normal);
			else
				direction = refract(unit_direction, payload.world_normal, ri);

			ray.setDir(direction);

			ray.setOrig(payload.world_position);
		}
		else if (current_material->Metallic) {
			ray.setDir(reflect(ray.getDirection(),
				payload.world_normal + (randomUnitSphereVec3(seed) * current_material->Roughness)));
		}
		else
		{//diffuse scattering
			ray.setDir(payload.world_normal + (randomUnitSphereVec3(seed)));
		}

		if (scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::DEBUGMODE)
		{
			if (scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::ALBEDO_DEBUG)
			{
				if (current_material->AlbedoTextureIndex < 0)
				{
					light = current_material->Albedo;
				}
				else
				{
					const Triangle* tri = (payload.primitiveptr);
					/*texture_sample_uv = {
						 payload.UVW.x * tri->vertex0.UV.x + payload.UVW.y * tri->vertex1.UV.x + payload.UVW.z * tri->vertex2.UV.x,
						  payload.UVW.x * tri->vertex0.UV.y + payload.UVW.y * tri->vertex1.UV.y + payload.UVW.z * tri->vertex2.UV.y
					};*/
					light = scenedata.DeviceTextureBufferPtr[current_material->AlbedoTextureIndex].getPixel(texture_sample_uv);
				}
			}
			if (scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::NORMAL_DEBUG)
				light = payload.world_normal;//debug normals
			if (scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::BARYCENTRIC_DEBUG)
				light = payload.UVW;//debug barycentric coords
			if (scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::UVS_DEBUG)
				light = { texture_sample_uv.x,texture_sample_uv.y,0 };//debug UV
			if (scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::MESHBVH_DEBUG)
			{
				light = { 0,0.1,0.1 };
				light += payload.color;
			}
			//break;
		}
		if (scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::DEBUGMODE)break;//break inside the 1st check?
	}

	if (scenedata.RenderSettings.gamma_correction &&
		(scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::NORMALMODE || scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::ALBEDO_DEBUG))
	{
		light = tonemapper(light, cam->exposure);
		light = gammaCorrection(light);
	}

	return light;
};