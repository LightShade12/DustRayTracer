#pragma once
#include <vector_types.h>

struct Vertex
{
	Vertex() = default;
	Vertex(float3 pos, float3 nrm, float2 uv) :position(pos), normal(nrm), UV(uv) {};
	float3 position;
	float3 normal;
	float2 UV;
};