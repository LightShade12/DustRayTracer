#pragma once
__device__ uint pcg_hash(uint input)
{
	uint state = input * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}
//0 to 1
__device__ float randomFloat(uint32_t& seed)
{
	seed = pcg_hash(seed);
	return (float)seed / (float)UINT32_MAX;
}

//-1 to 1
__device__ float3 randomUnitVec3(uint32_t& seed)
{
	return normalize(make_float3(
		randomFloat(seed) * 2.f - 1.f,
		randomFloat(seed) * 2.f - 1.f,
		randomFloat(seed) * 2.f - 1.f));
}

__device__ float3 randomUnitSphereVec3(uint32_t& seed) {
	while (true) {
		float3 p = randomUnitVec3(seed);
		float len = length(p);
		if ((len * len) < 1)
			return p;
	}
}
