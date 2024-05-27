#pragma once
#include <vector_types.h>
#include <float.h>

struct Bounds3f
{
	Bounds3f() = default;
	Bounds3f(float3 min, float3 max) :pMin(min), pMax(max) {};
	float3 pMin = { FLT_MAX,FLT_MAX, FLT_MAX };
	float3 pMax = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	float getSurfaceArea() const;//will return fltmin, fltmax if uninitialised
	float3 getCentroid() const;
};