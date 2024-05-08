#pragma once
#include "BVHNode.cuh"
#include <vector>

struct bin
{
	int prim_count = 0;
	Bounds3f bbox;
};

struct bvhPrimitive
{
	Triangle* primitive = nullptr;
	Bounds3f bounds;
};

/*
* bvh builder
*
* prerequisites:
* -triangle list(primitive lists)
* -triangle centroid
* -aabb intersection with t retval
*	tmin-tmax=0 then missed
*
* -preprocess
*	-cost function
* leaf nodes:
*	contain small list of triangles: 4-6
*	bvhs may overlap but not share a triangle
*
*/

class BVHBuilder
{
	BVHBuilder() = default;
	int bincount = 8;

	float3 getExtent(const BVHNode* node, float3& min_extent) //abs
	{
		float3 extent;
		float3 min = { FLT_MAX,FLT_MAX,FLT_MAX }, max = { -FLT_MAX,-FLT_MAX,-FLT_MAX };

		for (int primIdx = 0; primIdx < node->primitives_count; primIdx++)
		{
			const Triangle* tri = &(node->primitives[primIdx]);
			float3 positions[3] = { tri->vertex0.position, tri->vertex1.position,tri->vertex2.position };
			for (float3 pos : positions)
			{
				if (pos.x < min.x)min.x = pos.x;
				if (pos.y < min.y)min.y = pos.y;
				if (pos.z < min.z)min.z = pos.z;

				if (pos.x > max.x)max.x = pos.x;
				if (pos.y > max.y)max.y = pos.y;
				if (pos.z > max.z)max.z = pos.z;
			}
		}
		min_extent = min;
		extent = { max.x - min.x,max.y - min.y,max.z - min.z };
		return extent;
	};
	/*
	For each axis x,y,z:
	initialize buckets
	For each primitive p in node:
	b = compute_bucket (p.centroid)
	b. bbox.union(p.bbox) ;
	b.prim count++;
	For each of the B-1 possible partitioning planes
	Evaluate cost, keep track of lowest cost partition
	Recurse on lowest cost partition found (or make node a leaf if primcount is low enough)
	*/
	std::vector<bvhPrimitive>primitives_bounds;
	std::vector<Triangle>orderedPrimitives;

	Bounds3f getBounds(const Triangle& triangle)
	{
		float3 min = { FLT_MAX,FLT_MAX,FLT_MAX }, max = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
		float3 positions[3] = { triangle.vertex0.position, triangle.vertex1.position,triangle.vertex2.position };

		for (float3 pos : positions)
		{
			if (pos.x < min.x)min.x = pos.x;
			if (pos.y < min.y)min.y = pos.y;
			if (pos.z < min.z)min.z = pos.z;

			if (pos.x > max.x)max.x = pos.x;
			if (pos.y > max.y)max.y = pos.y;
			if (pos.z > max.z)max.z = pos.z;
		}

		return Bounds3f(min, max);
	}

	float3 getCentroid(Bounds3f bound)
	{
		return .5f * bound.pMin + .5f * bound.pMax;
	}

	void Preprocess(BVHNode* node) {
		for (int primIdx = 0; primIdx < node->primitives_count; primIdx++)
		{
			Triangle& tri = node->primitives[primIdx];
			Bounds3f bound = getBounds(tri);
			primitives_bounds.push_back({ &tri, bound });
		}
	}

	void BuildRecursive(BVHNode* root, std::vector<Triangle>primitives)
	{
	}

	void Build(BVHNode* node)
	{
		//x,y,z
		float3 minExtent;
		float3 extent = getExtent(node, minExtent);
		//for x
		bin left, right;
		for (int primIdx = 0; primIdx < node->primitives_count; primIdx++)
		{
		}
	}
};