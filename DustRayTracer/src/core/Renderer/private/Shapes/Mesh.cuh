#pragma once
#include "Triangle.cuh"

#include<vector_types.h>

#include <vector>

class Mesh
{
public:
	Mesh() = default;
	__host__ Mesh(const std::vector<float3> &positions, const std::vector<float3> &vertex_normals, const std::vector<float2>& vertex_UVs, uint32_t matidx = 0);

	__host__ void Cleanup();

	Bounds3f Bounds;
	Triangle* m_dev_triangles;//device ptr
	size_t m_trisCount = 0;
};