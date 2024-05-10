#pragma once
#include "core/Renderer/private/CudaMath/helper_math.cuh"
#include "core/Renderer/private/Shapes/Triangle.cuh"
#include "core/Renderer/private/Shapes/Bounds.cuh"

#include "core/Editor/Common/CudaCommon.cuh"

/// <summary>
/// Will reside on gpu memory.
/// Create ptr values FOR GPU, variables will be COPIED from cpu
/// </summary>
struct BVHNode
{
	BVHNode() = default;

	bool leaf = false;
	Bounds3f bbox;
	BVHNode* dev_child1 = nullptr;
	BVHNode* dev_child2 = nullptr;
	const Triangle** dev_primitives_ptrs = nullptr;//buffer of ptrs to a another buffer's content; buffer of triangle ptrs
	int primitives_count = 0;
	int rayint_cost = 2;
	int trav_cost = 1;

	float getArea()
	{
		float planex = 2 * (bbox.pMax.z - bbox.pMin.z) * (bbox.pMax.y - bbox.pMin.y);
		float planey = 2 * (bbox.pMax.z - bbox.pMin.z) * (bbox.pMax.x - bbox.pMin.x);
		float planez = 2 * (bbox.pMax.x - bbox.pMin.x) * (bbox.pMax.y - bbox.pMin.y);
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

		cudaFree(dev_primitives_ptrs);
		checkCudaErrors(cudaGetLastError());
	}

	//only use on built nodes
	int getCost()
	{
		if (leaf)
			return primitives_count * rayint_cost;
		else
		{
			int p1 = dev_child1->getArea() / getArea(), p2 = dev_child2->getArea() / getArea();//SAH
			int cost = trav_cost + (p1 * (dev_child1->getCost())) + (p2 * (dev_child2->getCost()));
			return cost;
		}
	}
};