#pragma once

#include "core/Renderer/private/Shapes/Triangle.cuh"
#include "core/Renderer/private/Shapes/Material.cuh"

#include <thrust/device_vector.h>

struct Scene
{
	thrust::device_vector<Triangle> m_Triangles;
	thrust::device_vector<Material> m_Material;
};