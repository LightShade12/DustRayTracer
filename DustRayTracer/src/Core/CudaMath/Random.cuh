#pragma once
#include "helper_math.cuh"

__device__ uint pcg_hash(uint input);

//0 to 1
__device__ float randomFloat(uint32_t& seed);

//-1 to 1
__device__ float3 randomUnitVec3(uint32_t& seed);

__device__ float3 randomUnitSphereVec3(uint32_t& seed);

__device__ float3 refract(const float3& uv, const float3& n, float etai_over_etat);