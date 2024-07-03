#include "math_ext.cuh"
#include "cuda_math.cuh"

__device__ float3 refract(const float3& uv, const float3& n, float etai_over_etat)
{
	float cos_theta = fmin(dot((-1.f * uv), n), 1.0f);
	float3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	float3 r_out_parallel = -sqrt(fabs(1.0f - dot(r_out_perp, r_out_perp))) * n;
	return r_out_perp + r_out_parallel;
}