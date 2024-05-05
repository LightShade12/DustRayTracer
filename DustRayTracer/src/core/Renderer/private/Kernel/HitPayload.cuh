#pragma once
#include <vector_types.h>
#include <cstdint>

struct Triangle;

struct HitPayload
{
	bool debug = false;
	float hit_distance = -1;
	float3 world_normal;
	float3 world_position;
	uint32_t object_idx;
	uint32_t triangle_idx;
	const Triangle* primitive=nullptr;
	float3 UVW;
};
