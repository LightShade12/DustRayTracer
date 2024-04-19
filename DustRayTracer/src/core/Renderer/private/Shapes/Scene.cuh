#pragma once
//#include "core/Common/Managed.hpp"

#include <vector_types.h>
#include <thrust/device_vector.h>

struct Sphere
{
	Sphere()=default;
	float3 Position = { 0,0,0 };
	float3 Albedo = { 1,1,1 };
	float Radius = 1.f;
};

struct Scene
{
	thrust::device_vector<Sphere> m_Spheres;
};