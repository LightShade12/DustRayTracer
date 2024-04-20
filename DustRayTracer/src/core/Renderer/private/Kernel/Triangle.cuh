#pragma once
#include <vector_types.h>

struct Triangle {
	float3 vertex0, vertex1, vertex2;
	uint32_t MaterialIdx = 0;
};