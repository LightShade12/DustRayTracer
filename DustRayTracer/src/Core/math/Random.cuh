#pragma once
#include "cuda_math.cuh"

__device__ uint pcg_hash(uint input);

//0 to 1
__device__ float randomFloat(uint32_t& seed);

//-1 to 1; cube
__device__ float3 randomUnitFloat3(uint32_t& seed);

//sphere
__device__ float3 randomUnitSphereFloat3(uint32_t& seed);

__device__ float2 randomFloat2Disk(uint32_t& seed);