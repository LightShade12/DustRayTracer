#include "ImportanceSampler.cuh"

#include "BRDF.cuh"
#include "Scene/Material.cuh"
#include "Common/physical_units.hpp"

#include "Core/CudaMath/Random.cuh"
#include "Core/CudaMath/helper_math.cuh"

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

__device__ float3 importanceSampleBRDF(float3 normal, float3 viewDir, const Material& material, uint32_t& seed, float& pdf, const SceneData& scene_data, float2 texture_uv) {
	float roughness = material.Roughness;
	float metallicity = material.Metallicity;
	float3 H{};
	float3 sampleDir;

	float random_value = randomFloat(seed);
	float2 xi = make_float2(randomFloat(seed), randomFloat(seed));//uniform rng sample

	//if (random_value < metallicity)
	if (false)
	{
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

	return normalize(sampleDir);
}