#pragma once
#include "BVHNode.cuh"

#include <thrust/universal_vector.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <memory>
#include <vector>

class BVHBuilder
{
public:
	BVHBuilder() = default;
	int m_BinCount = 8;
	int m_TargetLeafPrimitivesCount = 6;

	BVHNode* build(const thrust::universal_vector<Triangle>& primitives);

private:
	enum class PARTITION_AXIS
	{
		X_AXIS = 0,
		Y_AXIS,
		Z_AXIS
	};

	void recursiveBuild(BVHNode& node);

	//bin is in world space
	void binToNodes(BVHNode& left, BVHNode& right, float bin, PARTITION_AXIS axis, const Triangle** primitives_ptrs_buffer, size_t primitives_count);

	void makePartition(const Triangle** primitives_ptrs_buffer, size_t primitives_count, BVHNode& leftnode, BVHNode& rightnode);

	//TODO: change to getExtents
	float3 getAbsoluteExtent(const Triangle** primitives_ptrs_buffer, size_t primitives_count, float3& min_extent)
	{
		std::vector<Triangle>tempDevtoHostTriangles;
		for (size_t ptr_idx = 0; ptr_idx < primitives_count; ptr_idx++)
		{
			tempDevtoHostTriangles.push_back(*(primitives_ptrs_buffer[ptr_idx]));
		}
		return getAbsoluteExtent(tempDevtoHostTriangles.data(), primitives_count, min_extent);
	};

	//bounding box extents
	float3 getAbsoluteExtent(const Triangle* primitives_buffer, size_t primitives_count, float3& min_extent) //abs
	{
		float3 extent;
		float3 min = { FLT_MAX,FLT_MAX,FLT_MAX }, max = { -FLT_MAX,-FLT_MAX,-FLT_MAX };

		for (int primIdx = 0; primIdx < primitives_count; primIdx++)
		{
			const Triangle* tri = &(primitives_buffer[primIdx]);
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