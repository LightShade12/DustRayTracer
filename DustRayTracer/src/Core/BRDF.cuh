#pragma once
#include <vector_types.h>

class Material;
class SceneData;

__device__ float3 BRDF(float3 incoming_lightdir, float3 outgoing_viewdir, float3 normal, const SceneData& scene_data,
	const Material& material, const float2& texture_uv);
__device__ float G1_GGX_Schlick(float NoV, float roughness);
__device__ float G2_Smith(float3 wo, float3 wi, float3 normal, float roughness);
__device__ float3 fresnelSchlick(float cosTheta, float3 F0);
__device__ float D_GGX(float NoH, float roughness);