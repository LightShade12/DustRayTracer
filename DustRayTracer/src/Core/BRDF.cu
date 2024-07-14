#include "BRDF.cuh"

#include "Scene/Texture.cuh"
#include "Scene/Triangle.cuh"
#include "Scene/Mesh.cuh"
#include "BVH/BVHNode.cuh"
#include "Scene/Material.cuh"
#include "Scene/SceneData.cuh"

#include "Core/CudaMath/helper_math.cuh"
#include "Common/physical_units.hpp"
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
	float3 F = F0 + (1.0 - F0) * pow(1.0 - VoH, 5.0);
	return clamp(F, 0, 1);
}

__device__ float D_GGX(float NoH, float roughness) {
	float clamp_rough = fmaxf(roughness, 0.02f);
	float alpha = clamp_rough * clamp_rough;
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

//combined diffuse+specular brdf
__device__ float3 BRDF(float3 incoming_lightdir, float3 outgoing_viewdir, float3 normal, const SceneData& scene_data,
	const Material& material, const float2& texture_uv)
{
	float3 H = normalize(outgoing_viewdir + incoming_lightdir);

	float NoV = clamp(dot(normal, outgoing_viewdir), 0.0, 1.0);//TODO:change to maxf
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

	float3 f0 = make_float3(0.16 * (reflectance * reflectance));//f0=0.04 for most mats
	f0 = lerp(f0, baseColor, metallicity);
	float3 F = fresnelSchlick(VoH, f0);

	float3 spec = make_float3(0);

	if (NoL > 0 && VoH > 0) {
		float G = G_Smith(NoV, NoL, roughness);
		float D = D_GGX(NoH, roughness);
		spec = (F * D * G) / (4.0 * fmaxf(NoV, 0.0001) * NoL);//maybe clamp NOV?
	}

	float3 rhoD = baseColor;

	rhoD *= (1.0 - F);//F=Ks
	//rhoD *= disneyDiffuseFactor(NoV, NoL, VoH, roughness);	// optionally for less AO
	rhoD *= (1.0 - metallicity);

	float3 diff = rhoD / PI;
	diff *= NoL;//NoL is lambert falloff
	return diff + spec;
	//return diff;
}

__device__ float ImportanceSampleGGX_VNDF_PDF(float roughness, float3 N, float3 V, float3 L)
{
	float clamped_roughness = fmaxf(roughness, 0.01);
	float3 H = normalize(L + V);
	float NoH = clamp(dot(N, H), 0.f, 1.f);
	float VoH = clamp(dot(V, H), 0.f, 1.f);
	float alpha = clamped_roughness * clamped_roughness;
	float alpha2 = alpha * alpha;
	float NoH2 = NoH * NoH;

	//float b = (NoH2 * (alpha2 - 1.0) + 1.0);
	float b = NoH2 * alpha2 + (1 - NoH2);

	float D = alpha2 / (PI * (b * b));

	return (VoH > 0.0) ? D / (4.0 * VoH) : 0.0;
}