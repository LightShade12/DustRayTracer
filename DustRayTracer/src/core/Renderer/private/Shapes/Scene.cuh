#pragma once

#include "core/Renderer/private/Shapes/Material.cuh"
#include "core/Renderer/private/Shapes/Mesh.cuh"

#include <thrust/device_vector.h>

struct Scene
{
	thrust::device_vector<Mesh> m_Meshes;
	thrust::device_vector<Material> m_Material;

	~Scene();
};