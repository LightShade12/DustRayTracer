#pragma once
#include <vector_types.h>
class Material;
class SceneData;

__device__ float3 sampleGGX(float3 normal, float roughness, float2 xi);

__device__ float3 sampleCosineWeightedHemisphere(float3 normal, float2 xi);

//returns normalized direction
__device__ float3 importanceSampleBRDF(float3 normal, float3 viewDir, const Material& material, uint32_t& seed, float& pdf, const SceneData& scene_data, float2 texture_uv);