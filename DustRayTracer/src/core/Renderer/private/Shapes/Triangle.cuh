#pragma once
#include <vector_types.h>

struct Vertex
{
	float3 position;
};

struct Triangle {
	Vertex vertex0, vertex1, vertex2;
	float3 normal;
	uint32_t MaterialIdx = 0;
};