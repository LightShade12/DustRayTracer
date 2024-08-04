#pragma once
#include <cuda_runtime.h>
#include "Core/Ray.cuh"
#include <cstdint>

__host__ __device__ float deg2rad(float degree);

namespace DustRayTracer {
	struct CameraData
	{
		CameraData() = default;
		char name[32] = "unnamed";
		float exposure = 1.5;
		float vertical_fov_radians = deg2rad(60);
		float zfar = 0;
		float znear = 0;
		float aspect_ratio = 0;
		float defocus_cone_angle = 0;  // Variation angle of rays through each pixel
		float focus_dist = 10;

		float movement_speed = 10;

		//TODO: switch to glm
		float3 position = { 0,0,0 };
		float3 forward_vector = { 0,0,-1 };
		float3 upward_vector = { 0,1,0 };
		float3 right_vector = { 1,0,0 };
		uint32_t viewheight = 0, viewwidth = 0;//weird stuff
		__device__ Ray getRay(float2 screen_uv, float framewidth, float frameheight, uint32_t& seed) const;
	};
}