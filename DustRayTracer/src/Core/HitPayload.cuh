#pragma once
#include <vector_types.h>
#include <cstdint>
#include <float.h>

struct Triangle;

struct HitPayload
{
	bool front_face = true;//default: true
	bool debug = false;
	float hit_distance = -1;//default: -1
	float3 world_normal;
	float3 world_position;
	int triangle_idx = -1;
	//const Triangle* primitiveptr = nullptr;//default: nullptr
	float3 color = { 0,0,0 };
	float3 UVW;
};

//used for primitive dataretval
struct ShortHitPayload
{
	float hit_distance = -1;//default: -1
	float3 UVW;
	int triangle_idx = -1;
	//const Triangle* primitiveptr = nullptr;//default: nullptr
};