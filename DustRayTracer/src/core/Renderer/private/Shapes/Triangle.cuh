#pragma once
#include "core/Renderer/private/CudaMath/helper_math.cuh"
#include "Vertex.cuh"

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