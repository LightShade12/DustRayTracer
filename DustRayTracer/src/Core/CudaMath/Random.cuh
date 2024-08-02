#pragma once
#include "helper_math.cuh"

__device__ uint pcg_hash(uint input);

//0 to 1
__device__ float randomFloat(uint32_t& seed);

//-1 to 1
__device__ float3 randomUnitFloat3(uint32_t& seed);

__device__ float3 randomUnitSphereVec3(uint32_t& seed);

__device__ float2 random_in_unit_disk(uint32_t& seed);
