#pragma once
#include "Triangle.cuh"

#include<vector_types.h>

#include <vector>

class Mesh
{
public:
	Mesh() = default;

	std::string name;
	//int MaterialsIdx[16];
	//int MaterialCount = 0;
	int m_primitives_offset = -1;
	size_t m_trisCount = 0;
};