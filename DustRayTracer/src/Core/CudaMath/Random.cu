#include "Random.cuh"
//#include <cuda_runtime.h>
#include <vector_types.h>
#include "Core/CudaMath/helper_math.cuh"

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
/*
inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
	auto cos_theta = fmin(dot(-uv, n), 1.0);
	vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
	vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
	return r_out_perp + r_out_parallel;
}
*/
__device__ float3 refract(float3 i, float3 n, float eta)
{
	float cosi = dot(-i, n);
	float cost2 = 1.0f - eta * eta * (1.0f - cosi * cosi);
	float3 t = eta * i + ((eta * cosi - sqrt(abs(cost2))) * n);
	return t * (cost2 > 0);
}

__device__ float3 randomUnitFloat3(uint32_t& seed)
{
	return normalize(make_float3(
		randomFloat(seed) * 2.f - 1.f,
		randomFloat(seed) * 2.f - 1.f,
		randomFloat(seed) * 2.f - 1.f));
}

__device__ float3 randomUnitSphereVec3(uint32_t& seed)
{
	while (true) {
		float3 p = randomUnitFloat3(seed);
		float len = length(p);
		if ((len * len) < 1)
			return p;
	}
}

__device__ float2 random_in_unit_disk(uint32_t& seed) {
	while (true) {
		float2 p = make_float2(randomFloat(seed) * 2 - 1, randomFloat(seed) * 2 - 1);
		if (dot(p, p) < 1.0f)
			return p;
	}
}