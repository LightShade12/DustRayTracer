#pragma once
#include <vector_types.h>
struct Ray {
	Ray() = default;
	__device__ Ray(float3 orig, float3 dir) :origin(orig), direction(dir) {};
	float3 origin;
	float3 direction;
};