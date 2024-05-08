#pragma once
#include "core/Renderer/private/CudaMath/helper_math.cuh"
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
	Bounds3f() = default;
	Bounds3f(float3 min, float3 max) :pMin(min), pMax(max) {};
	float3 pMin = { FLT_MAX,FLT_MAX, FLT_MAX };
	float3 pMax = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
};

struct Triangle {
	Triangle() = default;
	Triangle(Vertex v0, Vertex v1, Vertex v2, float3 nrm, int mtlidx) :
		vertex0(v0), vertex1(v1), vertex2(v2), face_normal(nrm), MaterialIdx(mtlidx) {
		centroid = (vertex0.position + vertex1.position + vertex2.position) / 3;
	};

	float3 centroid;
	Vertex vertex0, vertex1, vertex2;
	float3 face_normal;
	int MaterialIdx = 0;
};