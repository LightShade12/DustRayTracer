#pragma once
#include "BVHNode.cuh"

#include <thrust/universal_vector.h>
#include <thrust/device_vector.h>

#include <vector>

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
public:
	BVHBuilder() = default;
	int bincount = 8;
	int target_leaf_prim_count = 4;

	float3 getExtent(const Triangle** primitives_ptr, size_t primitives_count, float3& min_extent)
	{
		std::vector<Triangle>temptriangles;
		for (size_t ptr_idx = 0; ptr_idx < primitives_count; ptr_idx++)
		{
			temptriangles.push_back(*(primitives_ptr[ptr_idx]));
		}
		return getExtent(temptriangles.data(), primitives_count, min_extent);
	};

	//bounding box extents
	float3 getExtent(const Triangle* primitives, size_t primitives_count, float3& min_extent) //abs
	{
		float3 extent;
		float3 min = { FLT_MAX,FLT_MAX,FLT_MAX }, max = { -FLT_MAX,-FLT_MAX,-FLT_MAX };

		for (int primIdx = 0; primIdx < primitives_count; primIdx++)
		{
			const Triangle* tri = &(primitives[primIdx]);
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
		//for (int primIdx = 0; primIdx < node->primitives_count; primIdx++)
		//{
		//	Triangle& tri = node->dev_primitives_ptrs[primIdx];
		//	Bounds3f bound = getBounds(tri);
		//	primitives_bounds.push_back({ &tri, bound });
		//}
	}

	enum class PARTITION_AXIS
	{
		X_AXIS = 0,
		Y_AXIS,
		Z_AXIS
	};

	//bin is in world space
	void binToNodes(BVHNode& left, BVHNode& right, float bin, PARTITION_AXIS axis,
		const Triangle* primitives, size_t primitives_count)
	{
		std::vector<const Triangle*>leftprim;
		std::vector<const Triangle*>rightprim;

		//sorting
		for (size_t primidx = 0; primidx < primitives_count; primidx++)
		{
			const Triangle* triangle = &(primitives[primidx]);

			switch (axis)
			{
			case BVHBuilder::PARTITION_AXIS::X_AXIS:
				if (triangle->centroid.x < bin)
				{
					leftprim.push_back(triangle);
				}
				else
				{
					rightprim.push_back(triangle);
				}
				break;
			case BVHBuilder::PARTITION_AXIS::Y_AXIS:

				if (triangle->centroid.y < bin)
				{
					leftprim.push_back(triangle);
				}
				else
				{
					rightprim.push_back(triangle);
				}
				break;
			case BVHBuilder::PARTITION_AXIS::Z_AXIS:
				if (triangle->centroid.z < bin)
				{
					leftprim.push_back(triangle);
				}
				else
				{
					rightprim.push_back(triangle);
				}
				break;
			default:
				break;
			}
		}

		left.primitives_count = leftprim.size();
		size_t buffersize = sizeof(const Triangle*) * leftprim.size();
		cudaMallocManaged(&(left.dev_primitives_ptrs), buffersize);
		cudaMemcpy(left.dev_primitives_ptrs, leftprim.data(), buffersize, cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());
		float3 leftminextent;
		float3 leftextent = getExtent(leftprim.data(), left.primitives_count, leftminextent);
		left.bbox = Bounds3f(leftminextent, leftminextent + leftextent);

		right.primitives_count = rightprim.size();
		buffersize = sizeof(const Triangle*) * rightprim.size();
		cudaMallocManaged(&(right.dev_primitives_ptrs), buffersize);
		cudaMemcpy(right.dev_primitives_ptrs, rightprim.data(), buffersize, cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());
		float3 rightminextent;
		float3 rightextent = getExtent(right.dev_primitives_ptrs, right.primitives_count, rightminextent);
		right.bbox = Bounds3f(rightminextent, rightminextent + rightextent);
	}

	void makePartition(const Triangle** primitives_ptr, size_t primitives_count, BVHNode& leftnode, BVHNode& rightnode)
	{
		std::vector<const Triangle*>temptris;
		for (size_t ptridx = 0; ptridx < primitives_count; ptridx++)
		{
			temptris.push_back((primitives_ptr[ptridx]));
		}
		makePartition(temptris.data(), primitives_count, leftnode, rightnode);
	}

	void makePartition(const Triangle* primitives, size_t primitives_count, BVHNode& leftnode, BVHNode& rightnode)
	{
		float lowestcost_partition_pt = 0;//best bin
		PARTITION_AXIS bestpartitionaxis;

		int lowestcost = INT_MAX;

		float3 minextent = { FLT_MAX,FLT_MAX,FLT_MAX };
		float3 extent = getExtent(primitives, primitives_count, minextent);
		Bounds3f parentbbox(minextent, minextent + extent);

		int traversal_cost = 1;

		BVHNode left, right;
		//for x
		std::vector<float>bins;//world space
		float deltapartition = extent.x / bincount;
		for (int i = 1; i < bincount; i++)
		{
			bins.push_back(minextent.x + (i * deltapartition));
		}
		for (float bin : bins)
		{
			binToNodes(left, right, bin, PARTITION_AXIS::X_AXIS, primitives, primitives_count);
			int cost = traversal_cost + ((left.getArea() / parentbbox.getArea()) * left.primitives_count * left.rayint_cost) +
				((right.getArea() / parentbbox.getArea()) * right.primitives_count * right.rayint_cost);
			if (cost < lowestcost)
			{
				bestpartitionaxis = PARTITION_AXIS::X_AXIS;
				lowestcost_partition_pt = bin;
			}
			left.Cleanup();
			right.Cleanup();
		}

		//for y
		bins.clear();
		deltapartition = extent.y / bincount;
		for (int i = 1; i < bincount; i++)
		{
			bins.push_back(minextent.y + (i * deltapartition));
		}
		for (float bin : bins)
		{
			binToNodes(left, right, bin, PARTITION_AXIS::Y_AXIS, primitives, primitives_count);
			int cost = traversal_cost + ((left.getArea() / parentbbox.getArea()) * left.primitives_count * left.rayint_cost) +
				((right.getArea() / parentbbox.getArea()) * right.primitives_count * right.rayint_cost);
			if (cost < lowestcost)
			{
				bestpartitionaxis = PARTITION_AXIS::Y_AXIS;
				lowestcost_partition_pt = bin;
			}
			left.Cleanup();
			right.Cleanup();
		}

		//for z
		bins.clear();
		deltapartition = extent.z / bincount;
		for (int i = 1; i < bincount; i++)
		{
			bins.push_back(minextent.z + (i * deltapartition));
		}
		for (float bin : bins)
		{
			binToNodes(left, right, bin, PARTITION_AXIS::Z_AXIS, primitives, primitives_count);
			int cost = traversal_cost + ((left.getArea() / parentbbox.getArea()) * left.primitives_count * left.rayint_cost) +
				((right.getArea() / parentbbox.getArea()) * right.primitives_count * right.rayint_cost);
			if (cost < lowestcost)
			{
				bestpartitionaxis = PARTITION_AXIS::Z_AXIS;
				lowestcost_partition_pt = bin;
			}
			left.Cleanup();
			right.Cleanup();
		}

		binToNodes(leftnode, rightnode, lowestcost_partition_pt,
			bestpartitionaxis, primitives, primitives_count);
	}

	void RecursiveBuild(BVHNode& node)
	{
		if (node.primitives_count <= target_leaf_prim_count)
		{
			node.leaf = true; return;
		}
		else
		{
			BVHNode* leftnode = new BVHNode();
			BVHNode* rightnode = new BVHNode();

			makePartition(node.dev_primitives_ptrs, node.primitives_count, *leftnode, *rightnode);

			RecursiveBuild(*leftnode);
			RecursiveBuild(*rightnode);

			cudaMallocManaged(&node.dev_child1, sizeof(BVHNode));
			cudaMemcpy(node.dev_child1, leftnode, sizeof(BVHNode), cudaMemcpyHostToDevice);
			delete leftnode;

			cudaMallocManaged(&node.dev_child2, sizeof(BVHNode));
			cudaMemcpy(node.dev_child2, rightnode, sizeof(BVHNode), cudaMemcpyHostToDevice);
			delete rightnode;
		}
	}

	BVHNode* Build(thrust::universal_vector<Triangle> primitives)
	{
		BVHNode* hostBVHroot = new BVHNode();

		float3 minextent;
		float3 extent = getExtent(thrust::raw_pointer_cast(primitives.data()),
			primitives.size(), minextent);//error prone
		hostBVHroot->bbox = Bounds3f(minextent, minextent + extent);

		//if leaf candidate
		if (primitives.size() <= target_leaf_prim_count)
		{
			hostBVHroot->leaf = true;
			std::vector<const Triangle*>temptris;
			for (size_t i = 0; i < primitives.size(); i++)
			{
				temptris.push_back(&(primitives[i]));
			}
			cudaMallocManaged(&(hostBVHroot->dev_primitives_ptrs), sizeof(const Triangle*) * primitives.size());
			cudaMemcpy(hostBVHroot->dev_primitives_ptrs, temptris.data(), sizeof(const Triangle*) * primitives.size(), cudaMemcpyHostToDevice);

			hostBVHroot->primitives_count = primitives.size();

			BVHNode* deviceBVHroot;
			cudaMallocManaged(&deviceBVHroot, sizeof(BVHNode));
			cudaMemcpy(deviceBVHroot, hostBVHroot, sizeof(BVHNode), cudaMemcpyHostToDevice);
			delete hostBVHroot;

			return deviceBVHroot;
		}

		BVHNode* left = new BVHNode();
		BVHNode* right = new BVHNode();

		makePartition(thrust::raw_pointer_cast(primitives.data()),
			primitives.size(), *(left), *(right));

		RecursiveBuild(*left);
		RecursiveBuild(*right);

		cudaMallocManaged(&hostBVHroot->dev_child1, sizeof(BVHNode));
		cudaMemcpy(hostBVHroot->dev_child1, left, sizeof(BVHNode), cudaMemcpyHostToDevice);
		delete left;

		cudaMallocManaged(&hostBVHroot->dev_child2, sizeof(BVHNode));
		cudaMemcpy(hostBVHroot->dev_child2, right, sizeof(BVHNode), cudaMemcpyHostToDevice);
		delete right;

		BVHNode* deviceBVHroot;
		cudaMallocManaged(&deviceBVHroot, sizeof(BVHNode));
		cudaMemcpy(deviceBVHroot, hostBVHroot, sizeof(BVHNode), cudaMemcpyHostToDevice);
		delete hostBVHroot;

		return deviceBVHroot;
	}
};