#pragma once
#include <vector_types.h>
#include <float.h>

struct Vertex
{
	Vertex() = default;
	Vertex(float3 pos, float3 nrm, float2 uv) :position(pos), normal(nrm), UV(uv) {};
	float3 position;
	float3 normal;
	float2 UV;
};

struct Bounds3f
{
	float3 pMin = { FLT_MAX,FLT_MAX, FLT_MAX };
	float3 pMax = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
};

struct Triangle {
	Triangle() = default;
	Triangle(Vertex v0, Vertex v1, Vertex v2, float3 nrm, uint32_t mtlidx) :
		vertex0(v0), vertex1(v1), vertex2(v2), face_normal(nrm), MaterialIdx(mtlidx) {};
	Vertex vertex0, vertex1, vertex2;
	float3 face_normal;
	uint32_t MaterialIdx = 0;
};