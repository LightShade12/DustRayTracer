#pragma once
#include <vector_types.h>
#include <cstdint>

struct HitPayload
{
	float hit_distance = -1;
	float3 world_normal;
	float3 world_position;
	uint32_t object_idx;
	float3 UVW;
};
