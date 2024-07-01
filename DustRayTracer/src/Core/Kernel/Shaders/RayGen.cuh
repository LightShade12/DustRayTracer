#pragma once
#include "Core/Kernel/TraceRay.cuh"

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

__device__ float3 sampleGGX(float3 normal, float roughness, float2 xi) {
	float alpha = roughness * roughness;

	float phi = 2.0f * PI * xi.x;
	float cosTheta = sqrtf((1.0f - xi.y) / (1.0f + (alpha * alpha - 1.0f) * xi.y));
	float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

	float3 H;
	H.x = sinTheta * cosf(phi);
	H.y = sinTheta * sinf(phi);
	H.z = cosTheta;

	float3 up = fabs(normal.z) < 0.999 ? make_float3(0.0, 0.0, 1.0) : make_float3(1.0, 0.0, 0.0);
	float3 tangent = normalize(cross(up, normal));
	float3 bitangent = cross(normal, tangent);

	return normalize(tangent * H.x + bitangent * H.y + normal * H.z);
}

__device__ float3 sampleCosineWeightedHemisphere(float3 normal, float2 xi) {
	// Generate a cosine-weighted direction in the local frame
	float phi = 2.0f * PI * xi.x;
	float cosTheta = sqrtf(xi.y);//TODO: might have to switch with sinTheta
	float sinTheta = sqrtf(1.0f - xi.y);

	float3 H;
	H.x = sinTheta * cosf(phi);
	H.y = sinTheta * sinf(phi);
	H.z = cosTheta;

	// Create an orthonormal basis (tangent, bitangent, normal)
	float3 up = fabs(normal.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
	float3 tangent = normalize(cross(up, normal));
	float3 bitangent = cross(normal, tangent);

	// Transform the sample direction from local space to world space
	return normalize(tangent * H.x + bitangent * H.y + normal * H.z);
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

__device__ float3 importanceSampleBRDF(float3 normal, float3 viewDir, const Material& material, uint32_t& seed, float& pdf, const SceneData& scene_data, float2 texture_uv) {
	float roughness = material.Roughness;
	float metallicity = material.Metallicity;
	float3 H{};
	float3 sampleDir;

	float random_value = randomFloat(seed);
	float2 xi = make_float2(randomFloat(seed), randomFloat(seed));//uniform rng sample

	if (random_value < metallicity) {
		// Metallic (Specular only)
		H = sampleGGX(normal, roughness, xi);
		sampleDir = reflect(-viewDir, H);
		pdf = D_GGX(dot(normal, H), roughness) * dot(normal, H) / (4.0f * dot(sampleDir, H));
	}
	else {
		// Non-metallic

		//diffuse
		sampleDir = sampleCosineWeightedHemisphere(normal, xi);
		pdf = dot(normal, sampleDir) * (1.0f / PI);
	}

	return sampleDir;
}



__device__ float3 BRDF(float3 incoming_lightdir, float3 outgoing_viewdir, float3 normal, const SceneData& scene_data,
	const Material& material, const float2& texture_uv) {
	float3 H = normalize(outgoing_viewdir + incoming_lightdir);

	float NoV = clamp(dot(normal, outgoing_viewdir), 0.0, 1.0);
	float NoL = clamp(dot(normal, incoming_lightdir), 0.0, 1.0);
	float NoH = clamp(dot(normal, H), 0.0, 1.0);
	float VoH = clamp(dot(outgoing_viewdir, H), 0.0, 1.0);

	float reflectance = material.Reflectance;
	float roughness = material.Roughness;
	float metallicity = material.Metallicity;
	float3 baseColor = material.Albedo;

	if (material.AlbedoTextureIndex >= 0)baseColor = scene_data.DeviceTextureBufferPtr[material.AlbedoTextureIndex].getPixel(texture_uv);

	//roughness-metallic texture
	if (material.RoughnessTextureIndex >= 0) {
		float3 col = scene_data.DeviceTextureBufferPtr[material.RoughnessTextureIndex].getPixel(texture_uv, true);
		roughness = col.y;
		metallicity = col.z;
	}

	if (scene_data.RenderSettings.UseMaterialOverride)
	{
		reflectance = scene_data.RenderSettings.OverrideMaterial.Reflectance;
		roughness = scene_data.RenderSettings.OverrideMaterial.Roughness;
		metallicity = scene_data.RenderSettings.OverrideMaterial.Metallicity;
		baseColor = scene_data.RenderSettings.OverrideMaterial.Albedo;
	}

	float3 f0 = make_float3(0.16 * (reflectance * reflectance));
	f0 = lerp(f0, baseColor, metallicity);

	float3 F = fresnelSchlick(VoH, f0);
	float D = D_GGX(NoH, roughness);
	float G = G_Smith(NoV, NoL, roughness);

	float3 spec = (F * D * G) / (4.0 * max(NoV, 0.001) * max(NoL, 0.001));

	float3 rhoD = baseColor;

	rhoD *= (1.0 - F);//F=Ks
	// optionally for less AO
	//rhoD *= disneyDiffuseFactor(NoV, NoL, VoH, roughness);

	rhoD *= (1.0 - metallicity);

	float3 diff = rhoD / PI;

	return diff + spec;
}

__device__ float3 normalMap(const Material& current_material,
	const Triangle* tri, float3 normal, const SceneData& scene_data,
	float2 texture_sample_uv) {
	float3 edge0 = tri->vertex1.position - tri->vertex0.position;
	float3 edge1 = tri->vertex2.position - tri->vertex0.position;
	float2 deltaUV0 = tri->vertex1.UV - tri->vertex0.UV;
	float2 deltaUV1 = tri->vertex2.UV - tri->vertex0.UV;
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

//Its called a Monte Carlo estimator

//TODO: maybe create a LaunchID struct instead of x,y?
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
	float3 cumulative_incoming_light_throughput = { 1,1,1 };//Transport operator
	int max_bounces = scenedata.RenderSettings.ray_bounce_limit;
	//int max_bounces = 1;

	HitPayload payload;
	//TODO: fix shadow at grazing angles issue
	//TODO: does russian roulette matter?
	//operator formulation for rendering equation
	for (int bounce_depth = 0; bounce_depth <= max_bounces; bounce_depth++)
	{
		payload = traceRay(ray, &scenedata);
		seed += bounce_depth;

		//SHADING------------------------------------------------------------
		//SKY SHADING----------------------------------
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
		const Material* current_material = &(scenedata.DeviceMaterialBufferPtr[payload.primitiveptr->material_idx]);
		const Triangle* tri = payload.primitiveptr;
		const float2 texture_sample_uv = payload.UVW.x * tri->vertex0.UV + payload.UVW.y * tri->vertex1.UV + payload.UVW.z * tri->vertex2.UV;
		//--------------------

		//TODO: fix smooth normals and triangle face normal situation
		//smooth shading
		payload.world_normal = normalize(payload.UVW.x * tri->vertex0.normal + payload.UVW.y * tri->vertex1.normal + payload.UVW.z * tri->vertex2.normal);
		if (!payload.front_face)payload.world_normal = -payload.world_normal;

		//normal map
		if (current_material->NormalTextureIndex >= 0) {
			payload.world_normal = normalMap(*current_material, tri,
				payload.world_normal, scenedata, texture_sample_uv);
		}

		//emission; currently ignores solid angle term for emitter
		outgoing_light += ((current_material->EmissionTextureIndex < 0) ? current_material->EmissiveColor :
			scenedata.DeviceTextureBufferPtr[current_material->EmissionTextureIndex].getPixel(texture_sample_uv)) * current_material->EmissiveScale * cumulative_incoming_light_throughput;

		float3 next_ray_origin = payload.world_position + (payload.world_normal * HIT_EPSILON);
		float pdf_eval;
		float3 viewdir = -1.f * (ray.getDirection());
		float3 next_ray_dir = importanceSampleBRDF(payload.world_normal, normalize(viewdir), *current_material, seed, pdf_eval, scenedata, texture_sample_uv);
		float3 lightdir = next_ray_dir;

		//for (int i = 0; i < scenedata.DeviceMeshLightsBufferSize; i++) {
		//	const Triangle* meshlight = &scenedata.DevicePrimitivesBuffer[scenedata.DeviceMeshLightsBufferPtr[i]];
		//	if (payload.primitiveptr == meshlight) {
		//		outgoing_light = meshlight->face_normal;
		//		break;
		//	}
		//}
		//break;

		// Direct light sampling
		const Triangle* meshlight = &scenedata.DevicePrimitivesBuffer[scenedata.DeviceMeshLightsBufferPtr[int(randomFloat(seed) * scenedata.DeviceMeshLightsBufferSize)]];
		//meshlight = &scenedata.DevicePrimitivesBuffer[2];
		if (payload.primitiveptr != meshlight) {
			float2 barycentric = { randomFloat(seed), randomFloat(seed) };
			//barycentric = {0.33,0.33};
			// Ensure that barycentric coordinates sum to 1
			if (barycentric.x + barycentric.y > 1.0f) {
				barycentric.x = 1.0f - barycentric.x;
				barycentric.y = 1.0f - barycentric.y;
			}
			// Calculate the target position on the mesh light
			float3 target = meshlight->vertex0.position * (1.0f - barycentric.x - barycentric.y) +
				meshlight->vertex1.position * barycentric.x +
				meshlight->vertex2.position * barycentric.y;

			float3 targetDir = normalize(target - next_ray_origin);

			Ray dlray(next_ray_origin, targetDir);
			dlray.interval = Interval(-1, FLT_MAX);
			//dlray.interval.max = length(target - next_ray_origin) + 0.01f;
			// Set the maximum interval to the distance to the target point minus a small epsilon
			HitPayload hp = traceRay(dlray, &scenedata);

			// Check if the ray hit the mesh light
			if (hp.primitiveptr != nullptr) {
				// Calculate the emission from the light
				float3 le = scenedata.DeviceMaterialBufferPtr[hp.primitiveptr->material_idx].EmissiveColor * 10;

				// Calculate the BRDF and other factors
				float3 brdf = BRDF(dlray.getDirection(), -1.f * ray.getDirection(), payload.world_normal, scenedata, *current_material, texture_sample_uv);
				float falloff = cosine_falloff_factor(dlray.getDirection(), payload.world_normal);
				float distanceSquared = dot(dlray.getOrigin() - hp.world_position, dlray.getOrigin() - hp.world_position);

				//outgoing_light += make_float3(0, 1, 0);
				// Accumulate the outgoing light
				outgoing_light += le * falloff * cumulative_incoming_light_throughput * brdf / distanceSquared;
			}
		}
		break;

		//shadow ray for sunlight
		if (scenedata.RenderSettings.enableSunlight && scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::NORMALMODE)
		{
			Ray sunray = Ray((next_ray_origin), (sunpos)+randomUnitFloat3(seed) * scenedata.RenderSettings.sun_size);
			if (!rayTest(sunray, &scenedata))
				outgoing_light += suncol * cumulative_incoming_light_throughput *
				BRDF(normalize(sunray.getDirection()), normalize(viewdir), payload.world_normal, scenedata, *current_material, texture_sample_uv) *
				cosine_falloff_factor(normalize(sunpos), payload.world_normal);
		}

		//prepare throughput for next bounce
		cumulative_incoming_light_throughput *=
			(BRDF(normalize(lightdir), normalize(viewdir), payload.world_normal, scenedata, *current_material, texture_sample_uv)
				* cosine_falloff_factor(lightdir, payload.world_normal)) / pdf_eval;

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
			break;//no bounce
		}
	}

	return outgoing_light;
};