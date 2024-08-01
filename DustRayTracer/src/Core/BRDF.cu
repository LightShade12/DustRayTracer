#include "BRDF.cuh"

#include "Scene/Texture.hpp"
#include "Scene/Triangle.cuh"
#include "Scene/Mesh.cuh"
#include "BVH/BVHNode.cuh"
#include "Scene/Material.cuh"
#include "Scene/SceneData.cuh"

#include "Core/CudaMath/helper_math.cuh"
#include "Core/CudaMath/physical_units.hpp"
#include <thrust/device_vector.h>

__device__ float fresnelSchlick90(float cosTheta, float F0, float F90) {
	float F = F0 + (F90 - F0) * pow(1.0 - cosTheta, 5.0);
	return clamp(F, 0.f, 1.f);
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

__device__ float3 BTDF(float3 incoming_lightdir, float3 outgoing_viewdir, float3 normal, const SceneData& scene_data,
	float3 albedo, float roughness, float3 F0, float metallicity, float trans, float ior) {
	return make_float3(1);
}

__device__ float3 specularBRDF(float3 incoming_lightdir, float3 outgoing_viewdir, float3 normal, const SceneData& scene_data,
	float3 albedo, float roughness, float3 F0, float metallicity) {
	float3 H = normalize(outgoing_viewdir + incoming_lightdir);
	float NoV = clamp(dot(normal, outgoing_viewdir), 0.0, 1.0);//TODO:change to maxf
	float NoL = clamp(dot(normal, incoming_lightdir), 0.0, 1.0);
	float NoH = clamp(dot(normal, H), 0.0, 1.0);
	float LoH = clamp(dot(incoming_lightdir, H), 0.0, 1.0);
	float VoH = clamp(dot(outgoing_viewdir, H), 0.0, 1.0);

	float3 F = fresnelSchlick(LoH, F0);

	float3 spec = make_float3(0);

	if (NoL > 0 && VoH > 0) {
		//float G = G_Smith(NoV, NoL, roughness);
		float G = G2_Smith(outgoing_viewdir, incoming_lightdir, normal, roughness);
		float D = D_GGX(NoH, roughness);
		spec = (F * D * G) / (4.0 * fmaxf(NoV, 0.0001) * NoL);//maybe clamp NOV?
	}

	return spec;
}

//cos_theta generally VoH
__device__ float3 fresnelRefractive(float cos_theta, float3 F0, float ni, float no) {
	float3 F = make_float3(0);
	float sin_thetao2 = powf(ni / no, 2) * (1 - powf(cos_theta, 2));
	float costhetao = sqrt(1 - sin_thetao2);
	if (no >= ni) {
		F = fresnelSchlick(cos_theta, F0);
	}
	else if (no < ni && sin_thetao2 < 1) {
		F = fresnelSchlick(costhetao, F0);
	}
	else if (no < ni && sin_thetao2 >= 1)
	{
		F = make_float3(1);
	}
}

__device__ float3 microFacetBRDF(float3 v, float3 l, float3 h, float3 N, float roughness) {
	float NoH = clamp(dot(N, h), 0.f, 1.f);
	float NoV = clamp(dot(N, v), 0.f, 1.f);
	float NoL = clamp(dot(N, l), 0.f, 1.f);

	//below expect squared roughness
	float D = D_GGX(NoH, roughness);
	float G = G2_Smith(v, l, N, roughness);

	float out = (D * G) / (4 * fmaxf(NoV, 0.0001) * NoL);

	return make_float3(out);
}

__device__ float3 microFacetBTDF(float3 v, float3 l, float3 N, float3 ht, float roughness, float ni, float no) {
	float NoL = clamp(dot(N, l), 0.f, 1.f);
	float NoV = clamp(dot(N, v), 0.f, 1.f);
	float LoH = clamp(dot(ht, l), 0.f, 1.f);
	float VoH = clamp(dot(ht, v), 0.f, 1.f);
	float NoH = clamp(dot(N, ht), 0.f, 1.f);

	float D = D_GGX(NoH, roughness);
	float G = G2_Smith(v, l, N, roughness);

	float no2 = no * no;
	float denom = ni * VoH + no * LoH;
	float denom2 = denom * denom;
	float prod = (LoH * VoH) / (NoL * NoV);
	float out = (prod * no2 * D * G) / denom2;

	return make_float3(out);
}

__device__ float3 diffuseBRDF(float3 albedo, float3 v, float3 l, float3 n) {
	float scalar_switch = (dot(v, n) * dot(l, n) > 0) ? 1 : 0;
	float3 out = make_float3(scalar_switch / PI);
}

//handle absorption, diffuse and transmission
__device__ float3 OpaqueDielectricBSDF(float3 V, float3 L, float3 H, float3 N, float3 albedo, float roughness, float IOR) {
	float VoH = clamp(dot(V, H), 0.f, 1.f);
	float3 F0 = make_float3(0.04);
	float3 B = diffuseBRDF(albedo, V, L, N);
	float3 Mr = microFacetBRDF(V, L, H, N, roughness);
	float3 F = fresnelSchlick(VoH, F0);
	float3 out = (albedo * B) + Mr * F;
}

__device__ float3 metallicBSDF(float3 V, float3 L, float3 H, float3 N, float3 rhoF0, float roughness) {
	float VoH = clamp(dot(V, H), 0.f, 1.f);
	float3 F = fresnelSchlick(VoH, rhoF0);
	float3 Mr = microFacetBRDF(V, L, H, N, roughness);
	return Mr * F;
}

//volumetric
__device__ float3 volumetricTransmissiveDielectricBSDF(float3 V, float3 L, float3 H, float3 N, float3 albedo, float roughness, float IOR) {
	float VoH = clamp(dot(V, H), 0.f, 1.f);
	float ni = 1;
	float no = IOR;
	float3 Ht = normalize((-V * ni) + (-L * no));
	float VoHt = clamp(dot(V, Ht), 0.f, 1.f);

	float3 F0 = make_float3(powf((1 - IOR) / (1 + IOR), 2));//assuming air as original medium
	float3 Fd = fresnelRefractive(VoH, F0, 1, IOR);
	float3 Fdr = fresnelRefractive(VoHt, F0, 1, IOR);
	float3 Mr = microFacetBRDF(V, L, H, N, roughness);
	float3 Mt = microFacetBTDF(V, L, N, Ht, roughness, ni, no);

	float3 out = (Mr * Fd) + (albedo * Mt * (1 - Fdr));
	return out;
}

__device__ float3 principledBSDF(float3 N, float3 V, float3 L, float3 albedo, float metallicity, float roughness, float transmission, float IOR) 
{
	float3 H = normalize(L + V);

	float3 M = metallicBSDF(V, L, H, N, albedo, roughness);
	float3 D = OpaqueDielectricBSDF(V, L, H, N, albedo, roughness, IOR);
	float3 T = volumetricTransmissiveDielectricBSDF(V, L, H, N, albedo, roughness, IOR);
	float3 out = (metallicity * M) + ((1 - metallicity) * (1 - transmission) * D) + ((1 - metallicity) * transmission * T);
	return out;
}

//combined diffuse+specular brdf; clamped roughness
__device__ float3 BRDF(float3 incoming_lightdir, float3 outgoing_viewdir, float3 normal, const SceneData& scene_data,
	float3 albedo, float roughness, float3 F0, float metallicity, float trans, float ior)
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
	rhoD *= (1 - trans);
	rhoD += trans;

	float3 diff = rhoD / PI;
	spec *= NoL;
	diff *= NoL;//NoL is lambert falloff //TODO:put this outside in rendering loop
	return diff + spec;
	//return diff;
}