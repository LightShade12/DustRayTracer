#pragma once
#include "core/Common/Managed.cuh"
#include "core/Renderer/private/CudaMath/helper_math.cuh"

//TODO: Camera InputProcess() which calls rotate and translate and cas mouseInput specific code

class Camera : public Managed
{
public:
	__host__ Camera(float3 pos = { 0,2,5 }) :m_Position(pos) { m_Right_dir = cross(m_Forward_dir, m_Up_dir); };

	__host__ void OnUpdate(float3 velocity, float delta);
	__host__ void Rotate(float4 delta_degrees);

	__device__ float3 GetRayDir(float2 _uv, float vfovdeg, float width, float height) const;

	__host__ float3 GetPosition() const { return m_Position; };

	void setMovementSpeed(float speed) { m_movement_speed = speed; };

	float m_movement_speed = 5;
	float3 m_Position = { 0,2,5 };
	float3 m_Forward_dir = { 0,0,-1 };
	float3 m_Up_dir = { 0,1,0 };
	float3 m_Right_dir = { 0,1,0 };
	uint viewheight = 0, viewwidth = 0;
};