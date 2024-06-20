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
	BVHNode* BuildIterative(thrust::universal_vector<Triangle>& primitives, thrust::device_vector<BVHNode>& bvh_nodes);
	BVHNode* BuildBVH(thrust::universal_vector<Triangle>& primitives, thrust::device_vector<BVHNode>& bvh_nodes);

private:
	enum class PartitionAxis
	{
		X_AXIS = 0,
		Y_AXIS,
		Z_AXIS
	};

	void recursiveBuild(BVHNode& node, thrust::device_vector<BVHNode>& bvh_nodes, std::vector<Triangle>& primitives);

	//bin is in world space
	void binToNodes(BVHNode& left, BVHNode& right, float bin, PartitionAxis axis, std::vector<Triangle>& primitives, size_t start_idx, size_t end_idx);
	void binToShallowNodes(BVHNode& left, BVHNode& right, float bin, PartitionAxis axis, std::vector<Triangle>& primitives, size_t start_idx, size_t end_idx);
	int costHeursitic(const BVHNode& left_node, const BVHNode& right_node, const Bounds3f& parent_bbox);
	void makePartition(std::vector<Triangle>& primitives, size_t start_idx, size_t end_idx, BVHNode& leftnode, BVHNode& rightnode);

public:
	int m_BinCount = 8;
	int m_TargetLeafPrimitivesCount = 6;
	int m_RayPrimitiveIntersectionCost = 2;
	int m_RayAABBIntersectionCost = 1;
};