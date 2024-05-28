#include "BVHBuilder.cuh"

BVHNode* BVHBuilder::build(const thrust::universal_vector<Triangle>& primitives)
{
	std::shared_ptr<BVHNode>hostBVHroot = std::make_shared<BVHNode>();

	printf("root prim count:%zu \n", primitives.size());

	float3 minextent;
	float3 extent = getAbsoluteExtent(thrust::raw_pointer_cast(primitives.data()),
		primitives.size(), minextent);//error prone
	hostBVHroot->m_BoundingBox = Bounds3f(minextent, minextent + extent);

	hostBVHroot->primitives_count = primitives.size();

	//if leaf candidate
	if (primitives.size() <= m_TargetLeafPrimitivesCount)
	{
		hostBVHroot->m_IsLeaf = true;
		std::vector<const Triangle*>DevToHostPrimitivePtrs;
		for (size_t i = 0; i < primitives.size(); i++)
		{
			DevToHostPrimitivePtrs.push_back(&(primitives[i]));
		}
		cudaMallocManaged(&(hostBVHroot->dev_primitive_ptrs_buffer), sizeof(const Triangle*) * DevToHostPrimitivePtrs.size());
		cudaMemcpy(hostBVHroot->dev_primitive_ptrs_buffer, DevToHostPrimitivePtrs.data(), sizeof(const Triangle*) * DevToHostPrimitivePtrs.size(), cudaMemcpyHostToDevice);

		BVHNode* deviceBVHroot;
		cudaMallocManaged(&deviceBVHroot, sizeof(BVHNode));
		cudaMemcpy(deviceBVHroot, hostBVHroot.get(), sizeof(BVHNode), cudaMemcpyHostToDevice);

		printf("made root leaf with %d prims\n", hostBVHroot->primitives_count);
		return deviceBVHroot;
	}

	std::shared_ptr<BVHNode>left = std::make_shared<BVHNode>();
	std::shared_ptr<BVHNode>right = std::make_shared<BVHNode>();

	//TODO: candidate for for_each loop
	thrust::host_vector<const Triangle*>dev_prim_ptrs;
	dev_prim_ptrs.reserve(primitives.size());
	for (size_t i = 0; i < primitives.size(); i++)
	{
		dev_prim_ptrs.push_back(&(primitives[i]));
	}

	makePartition(dev_prim_ptrs.data(),
		primitives.size(), *(left), *(right));

	recursiveBuild(*left);
	recursiveBuild(*right);

	cudaMallocManaged(&hostBVHroot->dev_child1, sizeof(BVHNode));
	cudaMemcpy(hostBVHroot->dev_child1, left.get(), sizeof(BVHNode), cudaMemcpyHostToDevice);

	cudaMallocManaged(&hostBVHroot->dev_child2, sizeof(BVHNode));
	cudaMemcpy(hostBVHroot->dev_child2, right.get(), sizeof(BVHNode), cudaMemcpyHostToDevice);

	BVHNode* deviceBVHroot;
	cudaMallocManaged(&deviceBVHroot, sizeof(BVHNode));
	cudaMemcpy(deviceBVHroot, hostBVHroot.get(), sizeof(BVHNode), cudaMemcpyHostToDevice);

	return deviceBVHroot;
}

void BVHBuilder::recursiveBuild(BVHNode& node)
{
	printf("recursive build, child node prim count: %d \n", node.primitives_count);
	if (node.primitives_count <= m_TargetLeafPrimitivesCount)
	{
		printf("made a leaf node with %d prims---------------\n", node.primitives_count);
		node.m_IsLeaf = true; return;
	}
	else
	{
		std::shared_ptr<BVHNode>leftnode = std::make_shared<BVHNode>();
		std::shared_ptr<BVHNode>rightnode = std::make_shared<BVHNode>();

		makePartition(node.dev_primitive_ptrs_buffer, node.primitives_count, *leftnode, *rightnode);
		cudaFree(node.dev_primitive_ptrs_buffer);
		//checkCudaErrors(cudaGetLastError());
		node.dev_primitive_ptrs_buffer = nullptr;

		recursiveBuild(*leftnode);
		recursiveBuild(*rightnode);

		cudaMallocManaged(&node.dev_child1, sizeof(BVHNode));
		cudaMemcpy(node.dev_child1, leftnode.get(), sizeof(BVHNode), cudaMemcpyHostToDevice);

		cudaMallocManaged(&node.dev_child2, sizeof(BVHNode));
		cudaMemcpy(node.dev_child2, rightnode.get(), sizeof(BVHNode), cudaMemcpyHostToDevice);
	}
}

void BVHBuilder::binToNodes(BVHNode& left, BVHNode& right, float bin, PARTITION_AXIS axis, const Triangle** primitives_ptrs_buffer, size_t primitives_count)
{
	std::vector<const Triangle*>left_prim_ptrs;
	std::vector<const Triangle*>right_prim_ptrs;

	//sorting
	//TODO: candidate for foreach
	for (size_t primidx = 0; primidx < primitives_count; primidx++)
	{
		const Triangle* triangle = (primitives_ptrs_buffer[primidx]);

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
	size_t buffersize = sizeof(const Triangle*) * left_prim_ptrs.size();
	cudaMallocManaged(&(left.dev_primitive_ptrs_buffer), buffersize);
	cudaMemcpy(left.dev_primitive_ptrs_buffer, left_prim_ptrs.data(), buffersize, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
	float3 leftminextent;
	float3 leftextent = getAbsoluteExtent(left_prim_ptrs.data(), left.primitives_count, leftminextent);//diff args good for error checking
	left.m_BoundingBox = Bounds3f(leftminextent, leftminextent + leftextent);

	right.primitives_count = right_prim_ptrs.size();
	buffersize = sizeof(const Triangle*) * right_prim_ptrs.size();
	cudaMallocManaged(&(right.dev_primitive_ptrs_buffer), buffersize);
	cudaMemcpy(right.dev_primitive_ptrs_buffer, right_prim_ptrs.data(), buffersize, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
	float3 rightminextent;
	float3 rightextent = getAbsoluteExtent(right.dev_primitive_ptrs_buffer, right.primitives_count, rightminextent);
	right.m_BoundingBox = Bounds3f(rightminextent, rightminextent + rightextent);
}

void BVHBuilder::binToShallowNodes(BVHNode& left, BVHNode& right, float bin, PARTITION_AXIS axis, const Triangle** primitives_ptrs_buffer, size_t primitives_count)
{
	std::vector<const Triangle*>left_prim_ptrs;
	std::vector<const Triangle*>right_prim_ptrs;

	//sorting
	//TODO: candidate for foreach
	for (size_t primidx = 0; primidx < primitives_count; primidx++)
	{
		const Triangle* triangle = (primitives_ptrs_buffer[primidx]);

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
	float3 leftextent = getAbsoluteExtent(left_prim_ptrs.data(), left.primitives_count, leftminextent);
	left.m_BoundingBox = Bounds3f(leftminextent, leftminextent + leftextent);

	right.primitives_count = right_prim_ptrs.size();
	float3 rightminextent;
	float3 rightextent = getAbsoluteExtent(right_prim_ptrs.data(), right.primitives_count, rightminextent);
	right.m_BoundingBox = Bounds3f(rightminextent, rightminextent + rightextent);
}

void BVHBuilder::makePartition(const Triangle** primitives_ptrs_buffer, size_t primitives_count, BVHNode& leftnode, BVHNode& rightnode)
{
	printf("partition input prim count:%zu \n", primitives_count);
	float lowestcost_partition_pt = 0;//best bin
	PARTITION_AXIS bestpartitionaxis{};

	int lowestcost = INT_MAX;

	float3 minextent = { FLT_MAX,FLT_MAX,FLT_MAX };
	float3 extent = getAbsoluteExtent(primitives_ptrs_buffer, primitives_count, minextent);
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
		binToShallowNodes(left, right, bin, PARTITION_AXIS::X_AXIS, primitives_ptrs_buffer, primitives_count);
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
		binToShallowNodes(left, right, bin, PARTITION_AXIS::Y_AXIS, primitives_ptrs_buffer, primitives_count);
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
		binToShallowNodes(left, right, bin, PARTITION_AXIS::Z_AXIS, primitives_ptrs_buffer, primitives_count);
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

	printf("made a partition, bin: %.3f, axis: %d, cost: %d\n", lowestcost_partition_pt, bestpartitionaxis, lowestcost);
	binToNodes(leftnode, rightnode, lowestcost_partition_pt,
		bestpartitionaxis, primitives_ptrs_buffer, primitives_count);
	printf("left node prim count:%d | right node prim count: %d\n", leftnode.primitives_count, rightnode.primitives_count);
}