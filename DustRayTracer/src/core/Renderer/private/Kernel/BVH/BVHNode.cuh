#pragma once
#include "core/Renderer/private/CudaMath/helper_math.cuh"
#include "core/Renderer/private/Shapes/Triangle.cuh"s

struct BVHNode
{
	bool leaf = false;
	Bounds3f bbox;
	BVHNode* child1 = nullptr;
	BVHNode* child2 = nullptr;
	Triangle* primitives = nullptr;//list of triangles
	int primitives_count = 0;
	int rayint_cost = 2;
	int trav_cost = 1;

	int getArea()
	{
		int planex = 2 * (bbox.pMax.z - bbox.pMin.z) * (bbox.pMax.y - bbox.pMin.y);
		int planey = 2 * (bbox.pMax.z - bbox.pMin.z) * (bbox.pMax.x - bbox.pMin.x);
		int planez = 2 * (bbox.pMax.x - bbox.pMin.x) * (bbox.pMax.y - bbox.pMin.y);
		return planex + planey + planez;
	}

	int getCost()
	{
		if (leaf)
			return primitives_count * rayint_cost;
		else
		{
			int p1 = child1->getArea() / getArea(), p2 = child2->getArea() / getArea();//SAH
			int cost = trav_cost + (p1 * (child1->getCost())) + (p2 * (child2->getCost()));
			return cost;
		}
	}
};