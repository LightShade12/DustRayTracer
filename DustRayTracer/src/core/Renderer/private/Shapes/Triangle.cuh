#pragma once
#include <vector_types.h>

struct Vertex
{
	Vertex() = default;
	Vertex(float3 pos, float3 nrm) :position(pos), normal(nrm) {};
	float3 position;
	float3 normal;
};

struct Triangle {
	Triangle() = default;
	Triangle(Vertex v0, Vertex v1, Vertex v2, float3 nrm, uint32_t mtlidx) :
		vertex0(v0), vertex1(v1), vertex2(v2), face_normal(nrm), MaterialIdx(mtlidx) {};
	Vertex vertex0, vertex1, vertex2;
	float3 face_normal;
	uint32_t MaterialIdx = 0;
};