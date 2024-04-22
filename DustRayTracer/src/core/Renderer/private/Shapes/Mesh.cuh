#pragma once
#include "Triangle.cuh"

#include<vector_types.h>

#include <vector>

class Mesh
{
public:
	Mesh(std::vector<float3> positions, std::vector<float3>normals, uint32_t matidx=0);
	~Mesh();
	Triangle* m_triangles;//device ptr
	size_t m_trisCount = 0;
};