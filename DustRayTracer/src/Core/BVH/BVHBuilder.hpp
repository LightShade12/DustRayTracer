#pragma once
#include "Core/Scene/HostScene.hpp"
#include "Core/BVH/BVHNode.cuh"
#include <memory>
#include <vector>

//TODO: spatial splits for 20% more improvement
class BVHBuilder
{
public:
	BVHBuilder() = default;
	BVHNode* BuildIterative(DustRayTracer::HostScene& scene);
	BVHNode* BuildBVH(DustRayTracer::HostScene& scene);

private:
	enum class PartitionAxis
	{
		X_AXIS = 0,
		Y_AXIS,
		Z_AXIS
	};

	void recursiveBuild(BVHNode& node, std::vector<BVHNode>& bvh_nodes,
		const DustRayTracer::HostScene& scene, std::vector<unsigned int>& primitive_indices);

	//bin is in world space
	void binToNodes(BVHNode& left, BVHNode& right, float bin, PartitionAxis axis,
		const DustRayTracer::HostScene& scene, std::vector<unsigned int>& primitives_indices, size_t start_idx, size_t end_idx);
	void binToShallowNodes(BVHNode& left, BVHNode& right, float bin, PartitionAxis axis,
		const DustRayTracer::HostScene& scene, std::vector<unsigned int>& primitives_indices, size_t start_idx, size_t end_idx);
	int costHeursitic(const BVHNode& left_node, const BVHNode& right_node, const Bounds3f& parent_bbox);
	void makePartition(const DustRayTracer::HostScene& scene, std::vector<unsigned int>& primitives_indices,
		size_t start_idx, size_t end_idx, BVHNode& leftnode, BVHNode& rightnode);

public:
	int m_BinCount = 8;
	int m_TargetLeafPrimitivesCount = 6;
	int m_RayPrimitiveIntersectionCost = 2;
	int m_RayAABBIntersectionCost = 1;
};