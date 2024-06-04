#pragma once
#include "Core/CudaMath/helper_math.cuh"

struct Ray {
	Ray() = default;
	__device__ Ray(float3 orig, float3 dir) :origin(orig), direction(dir) { invDir = (1.0f / dir); }
	__device__ void setDir(float3 new_dir) { direction = new_dir; invDir = 1.0f / new_dir; }
	float3 origin;
	float3 direction;
	float3 invDir;
};