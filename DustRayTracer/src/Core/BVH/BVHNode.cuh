#pragma once
#include "Core/Bounds.cuh"
#include "Core/Scene/Triangle.cuh"
#include "Core/CudaMath/helper_math.cuh"

#include "Editor/Common/CudaCommon.cuh"

#include <stdio.h>

/// <summary>
/// Will reside on gpu memory.
/// Create ptr values FOR GPU, variables will be COPIED from cpu
/// </summary>
class BVHNode
{
public:
	BVHNode() = default;

	Bounds3f m_BoundingBox;
	int left_start_idx = -1;//can be left child idx or tris start idx	//TODO:make them uint?
	int primitives_indices_count = 0;

	float getSurfaceArea() const
	{
		if (primitives_indices_count == 0)
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