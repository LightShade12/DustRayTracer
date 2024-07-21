#pragma once

#include <cuda_runtime.h>
#include <float.h>

//TODO: min interval doest work
class Interval
{
public:
	float min = -1, max = FLT_MAX;

	Interval() = default;

	__device__ Interval(float min, float max) :min(min), max(max) {};

	float size() const {
		return max - min;
	}

	bool contains(float x) const {
		return min <= x && x <= max;
	}

	__device__ bool surrounds(float x) const {
		return min < x && x < max;
	}

	~Interval() = default;

	static const Interval empty, universe;
};