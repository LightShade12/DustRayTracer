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
__device__ float3 refract(const float3& uv, const float3& n, float etai_over_etat)
{
	float cos_theta = fmin(dot((-1.f * uv), n), 1.0f);
	float3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	float3 r_out_parallel = -sqrt(fabs(1.0f - dot(r_out_perp, r_out_perp))) * n;
	return r_out_perp + r_out_parallel;
}

//Schlick fresnel
__device__ float reflectance(float cosine, float refraction_index)
{
	// Use Schlick's approximation for reflectance.
	float r0 = (1 - refraction_index) / (1 + refraction_index);
	r0 = r0 * r0;
	return r0 + (1 - r0) * powf((1 - cosine), 5.f);
}

__device__ float3 randomUnitVec3(uint32_t& seed)
{
	return normalize(make_float3(
		randomFloat(seed) * 2.f - 1.f,
		randomFloat(seed) * 2.f - 1.f,
		randomFloat(seed) * 2.f - 1.f));
}

__device__ float3 randomUnitSphereVec3(uint32_t& seed)
{
	while (true) {
		float3 p = randomUnitVec3(seed);
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