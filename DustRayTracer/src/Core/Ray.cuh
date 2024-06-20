#pragma once
#include "Core/Interval.cuh"
#include "Core/CudaMath/helper_math.cuh"

struct Ray {
	Ray() = default;
	__device__ Ray(float3 orig, float3 dir) :origin(orig), direction(dir) { inv_dir = (1.0f / dir); }

	__device__ void setDir(float3 new_dir) { direction = new_dir; inv_dir = 1.0f / new_dir; }
	__device__ void setOrig(float3 new_orig) { origin = new_orig; }

	__device__  inline const float3& getOrigin()const { return origin; }
	__device__  inline const float3& getDirection()const { return direction; }
	__device__  inline const float3& getInvDir()const { return inv_dir; }
public:
	Interval interval;
private:
	//interleaved order to apparently allow better 4-byte vector alignment in mem
	float3 inv_dir;
	//float min;
	float3 origin;
	//float max;
	float3 direction;
};