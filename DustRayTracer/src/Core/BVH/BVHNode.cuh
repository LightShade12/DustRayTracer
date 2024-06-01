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

	bool m_IsLeaf = false;
	Bounds3f m_BoundingBox;
	BVHNode* dev_child1 = nullptr;//left
	BVHNode* dev_child2 = nullptr;//right
	const Triangle** dev_primitive_ptrs_buffer = nullptr;//buffer of ptrs to a another buffer's content; buffer of triangle ptrs
	int primitives_count = 0;
	static const int rayint_cost = 2;
	static const int trav_cost = 1;

	float getSurfaceArea() const
	{
		if (primitives_count == 0)
			return 0;

		return m_BoundingBox.getSurfaceArea();
	}


	void Cleanup()
	{
		/*
		if (dev_child1 != nullptr) {
			dev_child1->Cleanup();
			cudaFree(dev_child1);
			checkCudaErrors(cudaGetLastError());
		}

		if (dev_child2 != nullptr) {
			dev_child2->Cleanup();
			cudaFree(dev_child2);
			checkCudaErrors(cudaGetLastError());
		}
		*/

		if (dev_primitive_ptrs_buffer != nullptr)
			cudaFree(dev_primitive_ptrs_buffer);
		checkCudaErrors(cudaGetLastError());
		//printf("node killed\n");
	}
};