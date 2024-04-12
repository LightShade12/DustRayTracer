#pragma once
#include "core/Common/Managed.hpp"
#include "core/Renderer/private/CudaMath/helper_math.hpp"

//TODO: Bring Camera controls to here

class Camera: public Managed
{
public:
	__host__ Camera() { m_Right_dir=cross(m_Forward_dir,m_Up_dir); };

	__device__ void Rotate(float2 mousedelta);
	__device__ void Translate(float3 movedir);

	__device__ void OnUpdate(float ts);
	__device__ void OnResize(uint width, uint height);
	__device__ float3 GetRayDir(float2 _uv, float vfovdeg, float width, float height);

	float3 m_Position = {0,0,5};
	float3 m_Forward_dir = { 0,0,-1 };
	float3 m_Up_dir = { 0,1,0 };
	float3 m_Right_dir = { 0,1,0 };
	uint viewheight = 0, viewwidth = 0;
};
