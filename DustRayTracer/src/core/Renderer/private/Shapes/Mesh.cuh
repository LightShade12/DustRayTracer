#pragma once
#include "Triangle.cuh"

#include<vector_types.h>

#include <vector>

class Mesh
{
public:
	Mesh() = default;
	__host__ Mesh(const std::vector<float3>& positions, const std::vector<float3>& vertex_normals,
		const std::vector<float2>& vertex_UVs, const std::vector<int>& prim_mat_idx);

	__host__ void Cleanup();

	Bounds3f Bounds;
	Triangle* m_dev_triangles;
	int m_primitives_offset = -1;
	size_t m_trisCount = 0;
};