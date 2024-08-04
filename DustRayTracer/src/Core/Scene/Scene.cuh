#pragma once
#include "HostScene.hpp"
#include "Core/BVH/BVHNode.cuh"
#include "Core/Scene/Triangle.cuh"
#include "Core/Scene/CameraData.cuh"
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>
#include <thrust/host_vector.h>

namespace DustRayTracer {
	struct Scene
	{
		Scene() = default;
		thrust::universal_vector<Mesh> m_MeshesBuffer;
		thrust::universal_vector<MaterialData> m_MaterialsBuffer;
		thrust::universal_vector<Texture>m_TexturesBuffer;
		thrust::universal_vector<BVHNode>m_BVHNodesBuffer;
		//lights vector
		thrust::universal_vector<int>m_TriangleLightsIndicesBuffer;
		thrust::universal_vector<unsigned int>m_BVHTrianglesIndicesBuffer;
		thrust::universal_vector<Triangle>m_TrianglesBuffer;
		thrust::universal_vector<DustRayTracer::CameraData>m_CamerasBuffer;

		void Cleanup();
	};
}