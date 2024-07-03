#include "Random.cuh"
#include "cuda_math.cuh"

#include <vector_types.h>

__device__ uint pcg_hash(uint input)
{
	uint state = input * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

__device__ float randomFloat(uint32_t& seed)
{
	seed = pcg_hash(seed);
	return (float)seed / (float)UINT32_MAX;
}

//cubic
__device__ float3 randomUnitFloat3(uint32_t& seed)
{
	return normalize(make_float3(
		randomFloat(seed) * 2.f - 1.f,
		randomFloat(seed) * 2.f - 1.f,
		randomFloat(seed) * 2.f - 1.f));
}

//sphere
__device__ float3 randomUnitSphereFloat3(uint32_t& seed)
{
	while (true) {
		float3 p = randomUnitFloat3(seed);
		float len = length(p);
		if ((len * len) < 1)
			return p;
	}
}

__device__ float2 randomFloat2Disk(uint32_t& seed) {
	while (true) {
		float2 p = make_float2(randomFloat(seed) * 2 - 1, randomFloat(seed) * 2 - 1);
		if (dot(p, p) < 1.0f)
			return p;
	}
}