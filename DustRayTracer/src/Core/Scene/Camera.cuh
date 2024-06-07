#pragma once
#include "Editor/Common/Managed.cuh"
#include "Core/CudaMath/helper_math.cuh"

/*
TODO: Camera InputProcess() which calls rotate and translate and has mouseInput specific code
add aspect ratio
*/

struct Ray;

class Camera : public Managed
{
public:
	__host__ Camera(float3 pos = { 0,2,5 }) :m_Position(pos) { m_Right_dir = cross(m_Forward_dir, m_Up_dir); };

	__host__ void OnUpdate(float3 velocity, float delta);
	__host__ void Rotate(float4 delta_degrees);

	__device__ Ray GetRay(float2 _uv, float width, float height, uint32_t& seed) const;

	//__device__ float3 GetRayDir(float2 _uv, float width, float height) const;

	__host__ float3 GetPosition() const { return m_Position; };

	void setMovementSpeed(float speed) { m_movement_speed = speed; };
private:
	__host__ __device__ float deg2rad(float degree);
public:
	//std::string name;
	float vfov_deg = deg2rad(60);//y_fov
	float zfar = 0;
	float znear = 0;
	float m_AspectRatio=0;

	float m_movement_speed = 10;
	float3 m_Position = { 0,2,5 };
	float3 m_Forward_dir = { 0,0,-1 };
	float3 m_Up_dir = { 0,1,0 };
	float3 m_Right_dir = { 0,1,0 };
	uint viewheight = 0, viewwidth = 0;
};