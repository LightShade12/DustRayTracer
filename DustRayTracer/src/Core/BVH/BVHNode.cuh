#pragma once
#include "Core/Bounds.cuh"
#include "Core/Scene/Triangle.cuh"
#include "Core/CudaMath/helper_math.cuh"

#include "Editor/Common/CudaCommon.cuh"

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

	void Cleanup()
	{
		/*if (dev_primitive_ptrs_buffer != nullptr)
			cudaFree(dev_primitive_ptrs_buffer);*/
		checkCudaErrors(cudaGetLastError());
		//printf("node killed\n");
	}
};