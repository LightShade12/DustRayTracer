#pragma once
#include "Core/CudaMath/helper_math.cuh"

__device__ static float3 uncharted2_tonemap_partial(float3 x)
{
	float A = 0.15f;
	float B = 0.50f;
	float C = 0.10f;
	float D = 0.20f;
	float E = 0.02f;
	float F = 0.30f;
	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

__device__ static float3 uncharted2_filmic(float3 v, float exposure)
{
	float exposure_bias = exposure;
	float3 curr = uncharted2_tonemap_partial(v * exposure_bias);

	float3 W = make_float3(11.2f);
	float3 white_scale = make_float3(1.0f) / uncharted2_tonemap_partial(W);
	return curr * white_scale;
}

__device__ static float3 toneMapping(float3 HDR_color, float exposure = 2.f) {
	float3 LDR_color = uncharted2_filmic(HDR_color, exposure);
	return LDR_color;
}

__device__ static float3 gammaCorrection(const float3 linear_color) {
	float3 gamma_space_color = { sqrtf(linear_color.x),sqrtf(linear_color.y) ,sqrtf(linear_color.z) };
	return gamma_space_color;
}