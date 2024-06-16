#include "BVHBuilder.cuh"
#include "Common/dbg_macros.hpp"
#include <stack>
#include <thrust/partition.h>
#include <algorithm>

//UNSABLE---------------------------------
// FIX BEFORE USING----------------------------------
// The build function initializes the root and uses a stack for iterative processing

BVHNode* BVHBuilder::buildIterative(thrust::universal_vector<Triangle>& primitives, thrust::device_vector<BVHNode>& bvh_nodes)
{
	//int MAX_STACK_SIZE = 2 * ((primitives.size() + primitives.size() / m_TargetLeafPrimitivesCount) / 2) - 1; // Adjust this value as needed

	BVHNode* hostBVHroot = new BVHNode();
	float3 minextent;
	float3 extent = getAbsoluteExtent(primitives, 0, primitives.size(), minextent);

	hostBVHroot->m_BoundingBox = Bounds3f(minextent, minextent + extent);
	hostBVHroot->primitive_start_idx = 0;
	hostBVHroot->primitives_count = primitives.size();
	printToConsole("root prim count:%zu \n", hostBVHroot->primitives_count);

	const int MAX_STACK_SIZE = 512; // Adjust this value as needed as per expected depth
	BVHNode* nodesToBeBuilt[MAX_STACK_SIZE]{};
	int stackPtr = 0;

	std::vector <Triangle> host_prims(primitives.begin(), primitives.end());
	std::vector <BVHNode> host_bvh_nodes;
	size_t nodecount=1024*100;
	host_bvh_nodes.reserve(nodecount);

	// If leaf candidate
	if (host_prims.size() <= m_TargetLeafPrimitivesCount)
	{
		hostBVHroot->m_IsLeaf = true;

		bvh_nodes.push_back(*hostBVHroot);
		delete hostBVHroot;
		printToConsole("made root leaf with %d prims\n", hostBVHroot->primitives_count);

		return thrust::raw_pointer_cast(&(bvh_nodes.back()));
	}

	// Static stack for iterative BVH construction
	nodesToBeBuilt[stackPtr++] = hostBVHroot;
	BVHNode* currentNode = nullptr;

	while (stackPtr > 0)
	{
		currentNode = nodesToBeBuilt[--stackPtr];

		// If the current node is a leaf candidate
		if (currentNode->primitives_count <= m_TargetLeafPrimitivesCount)
		{
			printToConsole("> made a leaf node with %d prims --------------->\n", currentNode->primitives_count);
			currentNode->m_IsLeaf = true;
			continue;
		}

		// Partition the current node
		BVHNode* leftNode = new BVHNode();
		BVHNode* rightNode = new BVHNode();

		makePartition(host_prims, currentNode->primitive_start_idx,
			currentNode->primitive_start_idx + currentNode->primitives_count, *leftNode, *rightNode);

		printToConsole("size before child1pushback: %zu\n", host_bvh_nodes.size());
		host_bvh_nodes.push_back(*leftNode); delete leftNode;
		currentNode->dev_child1_idx = host_bvh_nodes.size() - 1;
		printToConsole("size after child1pushback: %zu\n", host_bvh_nodes.size());

		host_bvh_nodes.push_back(*rightNode); delete rightNode;
		currentNode->dev_child2_idx = host_bvh_nodes.size() - 1;
		printToConsole("size after child2pushback: %zu\n", host_bvh_nodes.size());

		printToConsole("child1 idx %d\n", currentNode->dev_child1_idx);
		printToConsole("child2 idx %d\n", currentNode->dev_child2_idx);

		// Push the child nodes onto the stack
		nodesToBeBuilt[stackPtr++] = &host_bvh_nodes[currentNode->dev_child1_idx];
		nodesToBeBuilt[stackPtr++] = &host_bvh_nodes[currentNode->dev_child2_idx];
	}

	host_bvh_nodes.push_back(*hostBVHroot); delete hostBVHroot;
	host_bvh_nodes.shrink_to_fit();

	primitives = host_prims;
	bvh_nodes = host_bvh_nodes;

	return thrust::raw_pointer_cast(&(bvh_nodes.back()));
}

/*
TODO: sort triangle scene array into contiguous bvh leaf tri groups
Union of bbox in SAH bin computing
*/

//TODO:copy tris to host for bvh, let tris be dev_vector
BVHNode* BVHBuilder::build(thrust::universal_vector<Triangle>& primitives, thrust::device_vector<BVHNode>& bvh_nodes)
{
	std::shared_ptr<BVHNode>hostBVHroot = std::make_shared<BVHNode>();

	printToConsole("<--bvh build input tris count:%zu -->\n", primitives.size());
	std::vector<Triangle> host_prims(primitives.begin(), primitives.end());

	float3 minextent;
	float3 extent = getAbsoluteExtent(host_prims,
		0, host_prims.size(), minextent);
	hostBVHroot->m_BoundingBox = Bounds3f(minextent, minextent + extent);

	hostBVHroot->primitives_count = host_prims.size();

	//if leaf candidate
	if (host_prims.size() <= m_TargetLeafPrimitivesCount)
	{
		hostBVHroot->m_IsLeaf = true;
		hostBVHroot->primitive_start_idx = 0;

		bvh_nodes.push_back(*hostBVHroot);
		printToConsole("-----made RootNode leaf with %d prims-----\n", hostBVHroot->primitives_count);

		return thrust::raw_pointer_cast(&(bvh_nodes.back()));
	}

	std::shared_ptr<BVHNode>left = std::make_shared<BVHNode>();
	std::shared_ptr<BVHNode>right = std::make_shared<BVHNode>();

	makePartition(host_prims,
		0, host_prims.size(), *left, *right);

	recursiveBuild(*left, bvh_nodes, host_prims);
	recursiveBuild(*right, bvh_nodes, host_prims);

	bvh_nodes.push_back(*left);
	hostBVHroot->dev_child1_idx = bvh_nodes.size() - 1;

	bvh_nodes.push_back(*right);
	hostBVHroot->dev_child2_idx = bvh_nodes.size() - 1;

	primitives = host_prims;
	bvh_nodes.push_back(*hostBVHroot);

	return thrust::raw_pointer_cast(&(bvh_nodes.back()));
}

void BVHBuilder::recursiveBuild(BVHNode& node, thrust::device_vector<BVHNode>& bvh_nodes, std::vector<Triangle>& primitives)
{
	printToConsole("> recursive child build,input prim count: %d \n", node.primitives_count);

	if (node.primitives_count <= m_TargetLeafPrimitivesCount)
	{
		printToConsole("made a leaf node with %d prims---------------<\n", node.primitives_count);
		node.dev_child1_idx = -1, node.dev_child2_idx = -1;//redundant?
		node.m_IsLeaf = true; return;
	}
	else
	{
		std::shared_ptr<BVHNode>leftnode = std::make_shared<BVHNode>();//TODO: candidate for raw ptr
		std::shared_ptr<BVHNode>rightnode = std::make_shared<BVHNode>();

		makePartition(primitives, node.primitive_start_idx, node.primitive_start_idx + node.primitives_count, *leftnode, *rightnode);

		recursiveBuild(*leftnode, bvh_nodes, primitives);
		recursiveBuild(*rightnode, bvh_nodes, primitives);

		bvh_nodes.push_back(*leftnode);
		node.dev_child1_idx = bvh_nodes.size() - 1;

		bvh_nodes.push_back(*rightnode);
		node.dev_child2_idx = bvh_nodes.size() - 1;
	}
}

__host__ void BVHBuilder::binToNodes(BVHNode& left, BVHNode& right, float bin, PARTITION_AXIS axis, std::vector<Triangle>& primitives,
	size_t start_idx, size_t end_idx)
{
	//sorting
	std::vector<Triangle>::iterator it;
	/*auto it = thrust::partition(primitives.begin() + start_idx, primitives.begin() + end_idx,
		[bin](const Triangle& tri) {return tri.centroid.x < bin; });*/
	switch (axis)
	{
	case BVHBuilder::PARTITION_AXIS::X_AXIS:
		it = std::partition(primitives.begin() + start_idx, primitives.begin() + end_idx,
			[bin](const Triangle& tri) { return tri.centroid.x < bin; });
		break;
	case BVHBuilder::PARTITION_AXIS::Y_AXIS:
		it = std::partition(primitives.begin() + start_idx, primitives.begin() + end_idx,
			[bin](const Triangle& tri) { return tri.centroid.y < bin; });
		break;
	case BVHBuilder::PARTITION_AXIS::Z_AXIS:
		it = std::partition(primitives.begin() + start_idx, primitives.begin() + end_idx,
			[bin](const Triangle& tri) { return tri.centroid.z < bin; });
		break;
	default:
		break;
	}

	int idx = thrust::distance(primitives.begin(), it);
	left.primitive_start_idx = start_idx;
	left.primitives_count = idx - start_idx;
	float3 leftminextent;
	float3 leftextent = getAbsoluteExtent(primitives, left.primitive_start_idx,
		left.primitive_start_idx + left.primitives_count, leftminextent);
	left.m_BoundingBox = Bounds3f(leftminextent, leftminextent + leftextent);

	right.primitive_start_idx = idx;
	right.primitives_count = end_idx - idx;
	float3 rightminextent;
	float3 rightextent = getAbsoluteExtent(primitives, right.primitive_start_idx,
		right.primitive_start_idx + right.primitives_count, rightminextent);
	right.m_BoundingBox = Bounds3f(rightminextent, rightminextent + rightextent);
}

void BVHBuilder::binToShallowNodes(BVHNode& left, BVHNode& right, float bin, PARTITION_AXIS axis, std::vector<Triangle>& primitives,
	size_t start_idx, size_t end_idx)
{
	//can make a single vector and run partition
	std::vector<const Triangle*>left_prim_ptrs;
	std::vector<const Triangle*>right_prim_ptrs;

	for (size_t primidx = start_idx; primidx < end_idx; primidx++)
	{
		const Triangle* triangle = &(primitives[primidx]);

		switch (axis)
		{
		case BVHBuilder::PARTITION_AXIS::X_AXIS:
			if (triangle->centroid.x < bin)left_prim_ptrs.push_back(triangle);
			else right_prim_ptrs.push_back(triangle);
			break;
		case BVHBuilder::PARTITION_AXIS::Y_AXIS:
			if (triangle->centroid.y < bin)left_prim_ptrs.push_back(triangle);
			else right_prim_ptrs.push_back(triangle);
			break;
		case BVHBuilder::PARTITION_AXIS::Z_AXIS:
			if (triangle->centroid.z < bin)left_prim_ptrs.push_back(triangle);
			else right_prim_ptrs.push_back(triangle);
			break;
		default:
			break;
		}
	}

	left.primitives_count = left_prim_ptrs.size();
	float3 leftminextent;
	float3 leftextent = getAbsoluteExtent(left_prim_ptrs, 0, left_prim_ptrs.size(), leftminextent);
	left.m_BoundingBox = Bounds3f(leftminextent, leftminextent + leftextent);

	right.primitives_count = right_prim_ptrs.size();
	float3 rightminextent;
	float3 rightextent = getAbsoluteExtent(right_prim_ptrs, 0, right.primitives_count, rightminextent);
	right.m_BoundingBox = Bounds3f(rightminextent, rightminextent + rightextent);
}

void BVHBuilder::makePartition(std::vector<Triangle>& primitives, size_t start_idx, size_t end_idx,
	BVHNode& leftnode, BVHNode& rightnode)
{
	printToConsole("--->making partition, input prim count:%zu <---\n", end_idx - start_idx);
	float lowestcost_partition_pt = 0;//best bin
	PARTITION_AXIS bestpartitionaxis{};

	int lowestcost = INT_MAX;

	float3 minextent = { FLT_MAX,FLT_MAX,FLT_MAX };
	float3 extent = getAbsoluteExtent(primitives, start_idx, end_idx, minextent);
	Bounds3f parentbbox(minextent, minextent + extent);

	BVHNode left, right;

	//for x
	std::vector<float>bins;//world space
	bins.reserve(m_BinCount);
	float deltapartition = extent.x / m_BinCount;
	for (int i = 1; i < m_BinCount; i++)
	{
		bins.push_back(minextent.x + (i * deltapartition));
	}
	for (float bin : bins)
	{
		//printf("proc x bin %.3f\n", bin);
		binToShallowNodes(left, right, bin, PARTITION_AXIS::X_AXIS, primitives, start_idx, end_idx);
		int cost = BVHNode::trav_cost + ((left.getSurfaceArea() / parentbbox.getSurfaceArea()) * left.primitives_count * left.rayint_cost) +
			((right.getSurfaceArea() / parentbbox.getSurfaceArea()) * right.primitives_count * right.rayint_cost);
		if (cost < lowestcost)
		{
			lowestcost = cost;
			bestpartitionaxis = PARTITION_AXIS::X_AXIS;
			lowestcost_partition_pt = bin;
		}
		left.Cleanup();
		right.Cleanup();
	}

	//for y
	bins.clear();
	deltapartition = extent.y / m_BinCount;
	for (int i = 1; i < m_BinCount; i++)
	{
		bins.push_back(minextent.y + (i * deltapartition));
	}
	for (float bin : bins)
	{
		//printf("proc y bin %.3f\n", bin);
		binToShallowNodes(left, right, bin, PARTITION_AXIS::Y_AXIS, primitives, start_idx, end_idx);
		int cost = BVHNode::trav_cost + ((left.getSurfaceArea() / parentbbox.getSurfaceArea()) * left.primitives_count * left.rayint_cost) +
			((right.getSurfaceArea() / parentbbox.getSurfaceArea()) * right.primitives_count * right.rayint_cost);
		if (cost < lowestcost)
		{
			lowestcost = cost;
			bestpartitionaxis = PARTITION_AXIS::Y_AXIS;
			lowestcost_partition_pt = bin;
		}
		left.Cleanup();
		right.Cleanup();
	}

	//for z
	bins.clear();
	deltapartition = extent.z / m_BinCount;
	for (int i = 1; i < m_BinCount; i++)
	{
		bins.push_back(minextent.z + (i * deltapartition));
	}
	for (float bin : bins)
	{
		//printf("proc z bin %.3f\n", bin);
		binToShallowNodes(left, right, bin, PARTITION_AXIS::Z_AXIS, primitives, start_idx, end_idx);
		int cost = BVHNode::trav_cost + ((left.getSurfaceArea() / parentbbox.getSurfaceArea()) * left.primitives_count * left.rayint_cost) +
			((right.getSurfaceArea() / parentbbox.getSurfaceArea()) * right.primitives_count * right.rayint_cost);
		if (cost < lowestcost)
		{
			lowestcost = cost;
			bestpartitionaxis = PARTITION_AXIS::Z_AXIS;
			lowestcost_partition_pt = bin;
		}
		left.Cleanup();
		right.Cleanup();
	}

	printToConsole(">> made a partition, bin: %.3f, axis: %d, cost: %d <<---\n", lowestcost_partition_pt, bestpartitionaxis, lowestcost);
	binToNodes(leftnode, rightnode, lowestcost_partition_pt,
		bestpartitionaxis, primitives, start_idx, end_idx);
	printToConsole("left node prim count:%d | right node prim count: %d\n", leftnode.primitives_count, rightnode.primitives_count);
}