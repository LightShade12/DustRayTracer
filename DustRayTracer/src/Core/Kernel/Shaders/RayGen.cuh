#pragma once
#include "Core/Kernel/TraceRay.cuh"

#include "Core/BRDF.cuh"
#include "Core/ImportanceSampler.cuh"

#include "Common/physical_units.hpp"
#include "Core/CudaMath/helper_math.cuh"
#include "Core/CudaMath/Random.cuh"
#include "Core/CudaMath/cuda_mat.cuh"

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
	float vertical_gradient_factor = 0.5 * (1 + ray.getDirection()).y;//clamps to range 0-1
	float3 col1 = scenedata.RenderSettings.sky_color;
	float3 col2 = { 1,1,1 };
	float3 fcol = (float(1 - vertical_gradient_factor) * col2) + (vertical_gradient_factor * col1);
	fcol = { std::powf(fcol.x,2), std::powf(fcol.y,2) , std::powf(fcol.z,2) };
	return fcol;
}

__device__ static bool checknan(const float3& vec) {
	return isnan(vec.x) || isnan(vec.y) || isnan(vec.z);
}
__device__ static bool checkinf(const float3& vec) {
	return isinf(vec.x) || isinf(vec.y) || isinf(vec.z);
}

__device__ float3 clamp_output(float3 c)
{
	if ((checknan(c)) || (checkinf(c)))
		return make_float3(0);
	else
		return clamp(c, 0, 1000);
}

__device__ float3 normalMap(const Material& current_material,
	const Triangle* triangle, float3 normal, const SceneData& scene_data,
	float2 texture_sample_uv) {
	float3 edge0 = triangle->vertex1.position - triangle->vertex0.position;
	float3 edge1 = triangle->vertex2.position - triangle->vertex0.position;
	float2 deltaUV0 = triangle->vertex1.UV - triangle->vertex0.UV;
	float2 deltaUV1 = triangle->vertex2.UV - triangle->vertex0.UV;
	float invDet = 1.0f / (deltaUV0.x * deltaUV1.y - deltaUV1.x * deltaUV0.y);
	float3 tangent = invDet * (deltaUV1.y * edge0 - deltaUV0.y * edge1);
	//float3 bitangent = invDet * (-deltaUV1.x * edge0 + deltaUV0.x * edge1);
	float3 T = normalize(tangent);
	float3 N = normal;
	//T - normalize(T - dot(T, N) * N);
	float3 B = normalize(cross(N, T));

	Matrix3x3_d TBN(T, B, N);
	TBN = TBN.transpose();

	float3 alteredNormal = (scene_data.DeviceTextureBufferPtr[current_material.NormalTextureIndex].getPixel(texture_sample_uv, true) * 2 - 1);
	alteredNormal.x *= current_material.NormalMapScale;
	alteredNormal.y *= current_material.NormalMapScale;
	alteredNormal = normalize(alteredNormal);
	alteredNormal = normalize(TBN * alteredNormal);
	return alteredNormal;
}

//TODO: maybe create a LaunchID struct instead of x,y?
__device__ float3 rayGen(uint32_t x, uint32_t y, uint32_t max_x, uint32_t max_y,
	const Camera* device_camera, uint32_t frameidx, const SceneData scene_data) {
	float3 sunpos = make_float3(
		sin(scene_data.RenderSettings.sunlight_dir.x) * (1 - sin(scene_data.RenderSettings.sunlight_dir.y)),
		sin(scene_data.RenderSettings.sunlight_dir.y),
		cos(scene_data.RenderSettings.sunlight_dir.x) * (1 - sin(scene_data.RenderSettings.sunlight_dir.y))) * 100;
	float3 suncol = scene_data.RenderSettings.sunlight_color * scene_data.RenderSettings.sunlight_intensity;

	uint32_t seed = x + y * max_x;
	seed *= frameidx;

	float2 screen_uv = { (float(x) / max_x) ,(float(y) / max_y) };
	screen_uv = screen_uv * 2 - 1;
	Ray ray = device_camera->getRay(screen_uv, max_x, max_y, seed);
	ray.interval = Interval(-1, FLT_MAX);

	float3 outgoing_light = { 0,0,0 };
	float3 cumulative_incoming_light_throughput = { 1,1,1 };//Transport operator
	int max_bounces = scene_data.RenderSettings.ray_bounce_limit;
	float last_pdf_brdf_brdf = 1;
	float last_pdf_light_brdf = 1;

	HitPayload payload;
	//TODO: fix shadow at grazing angles issue
	//TODO: does russian roulette matter?
	//operator formulation for rendering equation
	for (int bounce_depth = 0; bounce_depth <= max_bounces; bounce_depth++)
	{
		payload = traceRay(ray, &scene_data);
		seed += bounce_depth;

		//SHADING--------------------------------------------------------------------------
		//SKY SHADING----------------------------------
		if (payload.primitiveptr == nullptr)
		{
			if (scene_data.RenderSettings.DebugMode == RendererSettings::DebugModes::MESHBVH_DEBUG &&
				scene_data.RenderSettings.RenderMode == RendererSettings::RenderModes::DEBUGMODE)outgoing_light = payload.color;
			else {
				float3 sky_integral_eval = skyModel(ray, scene_data) * scene_data.RenderSettings.sky_intensity;
				outgoing_light += sky_integral_eval * cumulative_incoming_light_throughput;
			}
			break;
		}

		//SURFACE SHADING-------------------------------------------------------------------
		const Material* current_material = &(scene_data.DeviceMaterialBufferPtr[payload.primitiveptr->material_idx]);
		const Triangle* hit_triangle = payload.primitiveptr;
		const float2 texture_sample_uv = payload.UVW.x * hit_triangle->vertex0.UV + payload.UVW.y * hit_triangle->vertex1.UV + payload.UVW.z * hit_triangle->vertex2.UV;
		//--------------------

		//TODO: fix smooth normals and triangle face normal situation
		//smooth shading
		payload.world_normal = normalize(payload.UVW.x * hit_triangle->vertex0.normal + payload.UVW.y * hit_triangle->vertex1.normal + payload.UVW.z * hit_triangle->vertex2.normal);
		if (!payload.front_face)payload.world_normal = -1.f * payload.world_normal;

		//normal map
		if (current_material->NormalTextureIndex >= 0) {
			payload.world_normal = normalMap(*current_material, hit_triangle,
				payload.world_normal, scene_data, texture_sample_uv);
		}

		//Indirect Emission;
		float pdf_light_brdf = 1;
		{
			float emitter_cosTheta = fabs(dot(hit_triangle->face_normal, -1.f * ray.getDirection()));
			float distanceSquared = payload.hit_distance * payload.hit_distance;
			float3 edge1 = hit_triangle->vertex1.position - hit_triangle->vertex0.position;
			float3 edge2 = hit_triangle->vertex2.position - hit_triangle->vertex0.position;
			float lightArea = 0.5f * length(cross(edge1, edge2));
			pdf_light_brdf = distanceSquared / lightArea * emitter_cosTheta;
		}
		float MIS_brdf_weight = (bounce_depth > 0 && scene_data.RenderSettings.useMIS) ?
			clamp(last_pdf_brdf_brdf / (last_pdf_brdf_brdf + pdf_light_brdf), 0.f, 1.f) : 1;

		outgoing_light += ((current_material->EmissionTextureIndex < 0) ? current_material->EmissiveColor :
			scene_data.DeviceTextureBufferPtr[current_material->EmissionTextureIndex].getPixel(texture_sample_uv))
			* current_material->EmissiveScale * cumulative_incoming_light_throughput * MIS_brdf_weight;

		float pdf_brdf_brdf = 1;
		float3 next_ray_origin = payload.world_position + (payload.world_normal * HIT_EPSILON);
		float3 viewdir = -1.f * ray.getDirection();
		ImportanceSampleData importancedata = importanceSampleBRDF(payload.world_normal, viewdir,
			*current_material, seed, pdf_brdf_brdf, cumulative_incoming_light_throughput, scene_data, texture_sample_uv);
		float3 next_ray_dir = importancedata.sampleDir;
		float3 lightdir = next_ray_dir;

		//Direct light sampling----------------------------------------------------
		const Triangle* emissive_triangle = nullptr;
		if (scene_data.DeviceMeshLightsBufferSize > 0)
			emissive_triangle = &scene_data.DevicePrimitivesBuffer[scene_data.DeviceMeshLightsBufferPtr[int(randomFloat(seed) * scene_data.DeviceMeshLightsBufferSize)]];
		if (payload.primitiveptr != emissive_triangle && scene_data.RenderSettings.useMIS)//it will crash
		{
			float2 barycentric = { randomFloat(seed), randomFloat(seed) };

			// Ensure that barycentric coordinates sum to 1
			if (barycentric.x + barycentric.y > 1.0f) {
				barycentric.x = 1.0f - barycentric.x;
				barycentric.y = 1.0f - barycentric.y;
			}
			// Calculate the triangle_sample_point position on the mesh light
			float3 triangle_sample_point = emissive_triangle->vertex0.position * (1.0f - barycentric.x - barycentric.y) +
				emissive_triangle->vertex1.position * barycentric.x +
				emissive_triangle->vertex2.position * barycentric.y;

			float3 nee_sample_dir = normalize(triangle_sample_point - next_ray_origin);
			float hit_distance_nee = length(triangle_sample_point - next_ray_origin);
			float3 surfnorm = emissive_triangle->face_normal;
			Ray shadow_ray(next_ray_origin, nee_sample_dir);

			shadow_ray.interval = Interval(-1, FLT_MAX);
			HitPayload shadowray_payload = traceRay(shadow_ray, &scene_data);

			//shadow_ray.interval = Interval(-1, dist - (dist * 0.5));
			//bool occluded = rayTest(shadow_ray, &scenedata);

			//if (!occluded)//guard against alpha test; pseudo visibility term
			if (shadowray_payload.primitiveptr == emissive_triangle)//guard against alpha test; pseudo visibility term
			{
				// Emission from the potential light triangle; handles non-light appropriately; pseudo visibility term
				float3 Le = scene_data.DeviceMaterialBufferPtr[emissive_triangle->material_idx].EmissiveColor *
					scene_data.DeviceMaterialBufferPtr[emissive_triangle->material_idx].EmissiveScale;

				float3 brdf_nee = BRDF(shadow_ray.getDirection(), -1.f * ray.getDirection(),
					payload.world_normal, scene_data, *current_material, texture_sample_uv);

				float reciever_cos_theta_nee = fmaxf(0, dot(shadow_ray.getDirection(), payload.world_normal));

				if (dot(emissive_triangle->face_normal, shadow_ray.getDirection()) > 0.f) surfnorm = -1.f * emissive_triangle->face_normal;

				float emitter_cos_theta_nee = fabs(dot(surfnorm, -1.f * shadow_ray.getDirection()));

				float3 edge1 = emissive_triangle->vertex1.position - emissive_triangle->vertex0.position;
				float3 edge2 = emissive_triangle->vertex2.position - emissive_triangle->vertex0.position;
				float light_area_nee = 0.5f * length(cross(edge1, edge2));

				float pdf_light_nee = (hit_distance_nee * hit_distance_nee) / (emitter_cos_theta_nee * light_area_nee);
				//float pdf_brdf_nee = dot(payload.world_normal, shadow_ray.getDirection()) * (1.0f / PI);//lambertian diffuse only
				float pdf_brdf_nee = getPDF(importancedata, viewdir, payload.world_normal,
					*current_material, scene_data, texture_sample_uv);

				float MIS_nee_weight = clamp(pdf_light_nee / (pdf_light_nee + pdf_brdf_nee), 0.f, 1.f);

				//TODO: the lightsbuffersize factor is probably linked to pdf; implement properly
				//TODO: consider specular component only if visible AND if misWeight is greater than 0
				outgoing_light += brdf_nee * Le * cumulative_incoming_light_throughput * /*reciever_cos_theta_nee **/
					scene_data.DeviceMeshLightsBufferSize * (MIS_nee_weight / pdf_light_nee);
			}
		}

		//shadow ray for sunlight
		if (scene_data.RenderSettings.enableSunlight && scene_data.RenderSettings.RenderMode == RendererSettings::RenderModes::NORMALMODE)
		{
			Ray sunray = Ray((next_ray_origin), (sunpos)+randomUnitFloat3(seed) * scene_data.RenderSettings.sun_size);
			sunray.interval = Interval(-1, FLT_MAX);
			if (!rayTest(sunray, &scene_data))
				outgoing_light += suncol * cumulative_incoming_light_throughput *
				BRDF(normalize(sunray.getDirection()), viewdir, payload.world_normal, scene_data, *current_material, texture_sample_uv)
				/** fmaxf(0, dot(normalize(sunpos), payload.world_normal)*/;
		}

		//prepare throughput for next bounce
		cumulative_incoming_light_throughput *=
			(BRDF(lightdir, viewdir, payload.world_normal, scene_data, *current_material, texture_sample_uv)
				/** fmaxf(0, dot(lightdir, payload.world_normal)))*/ / pdf_brdf_brdf);
		cumulative_incoming_light_throughput = clamp_output(cumulative_incoming_light_throughput);

		//BOUNCE RAY---------------------------------------------------------------------------------------
		if (checknan(cumulative_incoming_light_throughput) || checkinf(cumulative_incoming_light_throughput) ||
			checknan(next_ray_dir) || checkinf(next_ray_dir))break;

		last_pdf_brdf_brdf = pdf_brdf_brdf;
		last_pdf_light_brdf = pdf_light_brdf;
		ray.setOrig(next_ray_origin);
		ray.setDir(next_ray_dir);

		//Debug Views------------------------------------------------------------------------------------
		if (scene_data.RenderSettings.RenderMode == RendererSettings::RenderModes::DEBUGMODE)
		{
			switch (scene_data.RenderSettings.DebugMode)
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
			break;//no bounce
		}
	}

	return clamp_output(outgoing_light);
};