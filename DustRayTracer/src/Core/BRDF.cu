#include "BRDF.cuh"

#include "Scene/Texture.cuh"
#include "Scene/Triangle.cuh"
#include "Scene/Mesh.cuh"
#include "BVH/BVHNode.cuh"
#include "Scene/Material.cuh"
#include "Scene/SceneData.cuh"

#include "Core/CudaMath/helper_math.cuh"
#include "Core/CudaMath/physical_units.hpp"
#include <thrust/device_vector.h>

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

__device__ float3 fresnelSchlick(float VoH, float3 F0) {
	float3 F = F0 + (1.0 - F0) * powf(1.0 - VoH, 5.0);
	return clamp(F, 0, 1);
}

//clamped roughness
__device__ float D_GGX(float NoH, float roughness) {
	roughness = fmaxf(roughness, MAT_MIN_ROUGHNESS);//needed
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float NoH2 = NoH * NoH;
	float b = (NoH2 * (alpha2 - 1.0) + 1.0);//alt: NoH2 * alpha2 + (1 - NoH2)
	//float b = NoH2 * alpha2 + (1 - NoH2);//alt: NoH2 * alpha2 + (1 - NoH2)
	return alpha2 / (PI * (b * b));
}

//maybe wrong
__device__ float G1_GGX_Schlick(float NoV, float roughness) {
	float alpha = roughness * roughness;
	float k = alpha / 2.0;
	return NoV / (NoV * (1.0 - k) + k);
}
__device__ float G_Smith(float NoV, float NoL, float roughness) {
	return G1_GGX_Schlick(NoL, roughness) * G1_GGX_Schlick(NoV, roughness);
}

__device__ float G2_Smith(float3 wo, float3 wi, float3 normal, float roughness)
{
	float a2 = powf(roughness, 4);

	float dotNL = fmaxf(0, dot(normal, wi));
	float dotNV = fmaxf(0, dot(normal, wo));

	float denomA = dotNV * sqrtf(a2 + (1.0f - a2) * dotNL * dotNL);
	float denomB = dotNL * sqrtf(a2 + (1.0f - a2) * dotNV * dotNV);

	return 2.0f * dotNL * dotNV / (denomA + denomB);
}

//combined diffuse+specular brdf; clamped roughness
__device__ float3 BRDF(float3 incoming_lightdir, float3 outgoing_viewdir, float3 normal, const SceneData& scene_data,
	float3 albedo, float roughness, float3 F0, float metallicity)
{
	float3 H = normalize(outgoing_viewdir + incoming_lightdir);

	float NoV = clamp(dot(normal, outgoing_viewdir), 0.0, 1.0);//TODO:change to maxf
	float NoL = clamp(dot(normal, incoming_lightdir), 0.0, 1.0);
	float NoH = clamp(dot(normal, H), 0.0, 1.0);
	float LoH = clamp(dot(incoming_lightdir, H), 0.0, 1.0);
	float VoH = clamp(dot(outgoing_viewdir, H), 0.0, 1.0);

	if (scene_data.RenderSettings.UseMaterialOverride)
	{
		albedo = scene_data.RenderSettings.OverrideMaterial.Albedo;
		metallicity = scene_data.RenderSettings.OverrideMaterial.Metallicity;
		F0 = lerp(make_float3(0.16 * scene_data.RenderSettings.OverrideMaterial.Reflectance * scene_data.RenderSettings.OverrideMaterial.Reflectance), albedo, metallicity);
		roughness = scene_data.RenderSettings.OverrideMaterial.Roughness;
	}

	float3 F = fresnelSchlick(LoH, F0);

	float3 spec = make_float3(0);

	if (NoL > 0 && VoH > 0) {
		//float G = G_Smith(NoV, NoL, roughness);
		float G = G2_Smith(outgoing_viewdir, incoming_lightdir, normal, roughness);
		float D = D_GGX(NoH, roughness);
		spec = (F * D * G) / (4.0 * fmaxf(NoV, 0.0001) * NoL);//maybe clamp NOV?
	}

	float3 rhoD = albedo;

	//rhoD *= (1.0 - F);//F=Ks
	rhoD *= (1.f - spec);
	//rhoD *= disneyDiffuseFactor(NoV, NoL, VoH, roughness);	// optionally for less AO
	rhoD *= (1.0 - metallicity);

	float3 diff = rhoD / PI;
	spec *= NoL;
	diff *= NoL;//NoL is lambert falloff //TODO:put this outside in rendering loop
	return diff + spec;
	//return diff;
}