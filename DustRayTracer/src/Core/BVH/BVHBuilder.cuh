#pragma once
#include "BVHNode.cuh"

#include <thrust/universal_vector.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <memory>
#include <vector>

//TODO: spatial splits for 20% more improvement
class BVHBuilder
{
public:
	BVHBuilder() = default;
	int m_BinCount = 8;
	int m_TargetLeafPrimitivesCount = 6;

	//BVHNode* build(const thrust::universal_vector<Triangle>& primitives);

	BVHNode* buildIterative(thrust::universal_vector<Triangle>& primitives, thrust::device_vector<BVHNode>& bvh_nodes);
	BVHNode* build(thrust::universal_vector<Triangle>& primitives, thrust::device_vector<BVHNode>& bvh_nodes);
	void recursiveBuild(BVHNode& node, thrust::device_vector<BVHNode>& bvh_nodes, std::vector<Triangle>& primitives);
	//BVHNode* build(const thrust::host_vector<Triangle>& primitives, thrust::device_vector<BVHNode>& bvh_nodes);

private:
	enum class PARTITION_AXIS
	{
		X_AXIS = 0,
		Y_AXIS,
		Z_AXIS
	};

	//void recursiveBuild(BVHNode& node, thrust::device_vector<BVHNode>& bvh_nodes);

	//bin is in world space
	//void binToNodes(BVHNode& left, BVHNode& right, float bin, PARTITION_AXIS axis, const Triangle** primitives_ptrs_buffer, size_t primitives_count);
	__host__ void binToNodes(BVHNode& left, BVHNode& right, float bin, PARTITION_AXIS axis, std::vector<Triangle>& primitives, size_t start_idx, size_t end_idx);
	void binToShallowNodes(BVHNode& left, BVHNode& right, float bin, PARTITION_AXIS axis, std::vector<Triangle>& primitives, size_t start_idx, size_t end_idx);
	//void binToShallowNodes(BVHNode& left, BVHNode& right, float bin, PARTITION_AXIS axis, const Triangle** primitives_ptrs_buffer, size_t primitives_count);

	void makePartition(std::vector<Triangle>& primitives, size_t start_idx, size_t end_idx, BVHNode& leftnode, BVHNode& rightnode);

	//void makePartition(const Triangle** primitives_ptrs_buffer, size_t primitives_count, BVHNode& leftnode, BVHNode& rightnode);

	//TODO: change to getExtents
	//bounding box extents
	float3 getAbsoluteExtent(const thrust::universal_vector<Triangle>& primitives, size_t start_idx, size_t end_idx, float3& min_extent) //abs
	{
		float3 extent;
		float3 min = { FLT_MAX,FLT_MAX,FLT_MAX }, max = { -FLT_MAX,-FLT_MAX,-FLT_MAX };

		for (int primIdx = start_idx; primIdx < end_idx; primIdx++)
		{
			const Triangle* tri = &(primitives[primIdx]);
			float3 positions[3] = { tri->vertex0.position, tri->vertex1.position, tri->vertex2.position };
			for (float3 pos : positions)
			{
				min.x = fminf(min.x, pos.x);
				min.y = fminf(min.y, pos.y);
				min.z = fminf(min.z, pos.z);

				max.x = fmaxf(max.x, pos.x);
				max.y = fmaxf(max.y, pos.y);
				max.z = fmaxf(max.z, pos.z);
			}
		}
		min_extent = min;
		extent = { max.x - min.x,max.y - min.y,max.z - min.z };
		return extent;
	};
	float3 getAbsoluteExtent(const std::vector<const Triangle*>& primitives, size_t start_idx, size_t end_idx, float3& min_extent) //abs
	{
		float3 extent;
		float3 min = { FLT_MAX,FLT_MAX,FLT_MAX }, max = { -FLT_MAX,-FLT_MAX,-FLT_MAX };

		for (int primIdx = start_idx; primIdx < end_idx; primIdx++)
		{
			const Triangle* tri = (primitives[primIdx]);
			float3 positions[3] = { tri->vertex0.position, tri->vertex1.position, tri->vertex2.position };
			for (float3 pos : positions)
			{
				min.x = fminf(min.x, pos.x);
				min.y = fminf(min.y, pos.y);
				min.z = fminf(min.z, pos.z);

				max.x = fmaxf(max.x, pos.x);
				max.y = fmaxf(max.y, pos.y);
				max.z = fmaxf(max.z, pos.z);
			}
		}
		min_extent = min;
		extent = { max.x - min.x,max.y - min.y,max.z - min.z };
		return extent;
	};
};