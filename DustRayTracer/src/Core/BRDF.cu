#include "BRDF.cuh"
#include "Scene/Material.cuh"
#include "Scene/Texture.cuh"
#include "Scene/Triangle.cuh"
#include "Scene/Mesh.cuh"
#include "BVH/BVHNode.cuh"
#include "math/Random.cuh"
#include "Scene/SceneData.cuh"
#include "Common/physical_units.hpp"
#include "Scene/RendererSettings.h"

__device__ float D_GGX(float NoH, float roughness) {
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float NoH2 = NoH * NoH;
	float b = (NoH2 * (alpha2 - 1.0) + 1.0);
	return alpha2 * (1 / PI) / (b * b);
}

__device__ float3 fresnelSchlick(float cosTheta, float3 F0) {
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
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

__device__ float3 BRDF::evaluateContribution(const float3& wi, const float3& wo, const float3& normal, float2 texcoords)
{
	float3 H = normalize(wo + wi);

	float NoV = clamp(dot(normal, wo), 0.0, 1.0);
	float NoL = clamp(dot(normal, wi), 0.0, 1.0);
	float NoH = clamp(dot(normal, H), 0.0, 1.0);
	float VoH = clamp(dot(wo, H), 0.0, 1.0);

	float reflectance = m_material->Reflectance;
	float roughness = m_material->Roughness;
	float metallicity = m_material->Metallicity;
	float3 baseColor = m_material->Albedo;

	if (m_material->AlbedoTextureIndex >= 0)baseColor = m_scene_data->DeviceTextureBufferPtr[m_material->AlbedoTextureIndex].getPixel(texcoords);

	//roughness-metallic texture
	if (m_material->ORMTextureIndex >= 0) {
		float3 col = m_scene_data->DeviceTextureBufferPtr[m_material->ORMTextureIndex].getPixel(texcoords, true);
		roughness = col.y;
		metallicity = col.z;
	}

	if (m_scene_data->RenderSettings.UseMaterialOverride)
	{
		reflectance = m_scene_data->RenderSettings.OverrideMaterial.Reflectance;
		roughness = m_scene_data->RenderSettings.OverrideMaterial.Roughness;
		metallicity = m_scene_data->RenderSettings.OverrideMaterial.Metallicity;
		baseColor = m_scene_data->RenderSettings.OverrideMaterial.Albedo;
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

__device__ float3 BRDF::importanceSample(const float3& wo, const float3& normal, uint32_t& seed, float& pdf)
{
	float roughness = m_material->Roughness;
	float metallicity = m_material->Metallicity;
	float3 H{};
	float3 sampleDir;

	float random_value = randomFloat(seed);
	float2 xi = make_float2(randomFloat(seed), randomFloat(seed));//uniform rng sample

	if (random_value < metallicity) {
		// Metallic (Specular only)
		H = sampleGGX(normal, roughness, xi);
		sampleDir = reflect(-1.f * wo, H);
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