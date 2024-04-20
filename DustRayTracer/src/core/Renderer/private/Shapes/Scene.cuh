#pragma once
//#include "core/Common/Managed.hpp"
#include "core/Renderer/private/Kernel/Triangle.cuh"

#include <vector_types.h>
#include <thrust/device_vector.h>

struct Sphere
{
	Sphere()=default;
	float3 Position = { 0,0,0 };
	float Radius = 1.f;
	uint32_t MaterialIndex = 0;
};

struct Material 
{
	float3 Albedo = { 1,1,1 };
	float Roughness = 0.8f;
};

struct Scene
{
	thrust::device_vector<Triangle> m_Triangles;
	thrust::device_vector<Material> m_Material;
};