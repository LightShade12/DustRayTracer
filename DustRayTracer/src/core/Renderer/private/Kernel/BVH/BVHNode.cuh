#pragma once
#include "core/Renderer/private/Shapes/Bounds.cuh"
#include "core/Renderer/private/Shapes/Triangle.cuh"
#include "core/Renderer/private/CudaMath/helper_math.cuh"

#include "core/Editor/Common/CudaCommon.cuh"

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
	const int rayint_cost = 2;
	const int trav_cost = 1;

	float getSurfaceArea() const
	{
		if (primitives_count == 0)
			return 0;

		float planex = 2 * (m_BoundingBox.pMax.z - m_BoundingBox.pMin.z) * (m_BoundingBox.pMax.y - m_BoundingBox.pMin.y);
		float planey = 2 * (m_BoundingBox.pMax.z - m_BoundingBox.pMin.z) * (m_BoundingBox.pMax.x - m_BoundingBox.pMin.x);
		float planez = 2 * (m_BoundingBox.pMax.x - m_BoundingBox.pMin.x) * (m_BoundingBox.pMax.y - m_BoundingBox.pMin.y);
		return planex + planey + planez;
	}


	void Cleanup()
	{
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

		cudaFree(dev_primitive_ptrs_buffer);
		checkCudaErrors(cudaGetLastError());
		//printf("node killed\n");
	}
};