#pragma once
#include "Core/Bounds.cuh"
#include "Core/CudaMath/helper_math.cuh"

#include <stdio.h>

// Will reside on gpu memory.
class BVHNode
{
public:
	BVHNode() = default;

	Bounds3f m_BoundingBox;
	int left_child_or_triangle_indices_start_idx = -1;	//TODO:make them uint?
	int triangle_indices_count = 0;

	float getSurfaceArea() const
	{
		if (triangle_indices_count == 0)
			return 0;

		return m_BoundingBox.getSurfaceArea();
	}

	//TODO: get rifd of this
	void Cleanup()
	{
	}
};