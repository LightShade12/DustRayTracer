#pragma once
#include <vector_types.h>

class Material;
class SceneData;

struct ImportanceSampleData {
	__device__ ImportanceSampleData() = default;
	__device__ ImportanceSampleData(float3 sampledir, bool is_specular)
		:sampleDir(sampledir), specular(is_specular) {};
	float3 sampleDir;
	bool specular;
};

__device__ float3 sampleCosineWeightedHemisphere(float3 normal, float2 xi);

__device__ float getPDF(float3 in_dir, bool specular, float3 out_dir, float3 normal, float roughness);

//returns normalized direction
__device__ ImportanceSampleData importanceSampleBRDF(float3 normal, float3 viewDir, const Material& material, uint32_t& seed, float& pdf,
	float3& throughput, const SceneData& scene_data, float2 texture_uv);