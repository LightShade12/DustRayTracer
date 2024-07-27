#include "HostScene.hpp"
#include "Scene.cuh"
//TODO: apparently this must exist on the device: its the standard
//#define STB_IMAGE_IMPLEMENTATION
//#include "Core/Common/dbg_macros.hpp"
#include "Core/Common/CudaCommon.cuh"

#include "stb_image.h"
#include <tiny_gltf.h>

namespace DustRayTracer {
	void Scene::Cleanup()
	{
		cudaDeviceSynchronize();

		thrust::host_vector<BVHNode>nodes = m_BVHNodesBuffer;

		for (BVHNode node : nodes) {
			//printf("node freed\n");
			node.Cleanup();
		}

		checkCudaErrors(cudaGetLastError());

		for (Texture texture : m_TexturesBuffer)
		{
			texture.Cleanup();
		}
		checkCudaErrors(cudaGetLastError());

#ifdef DEBUG
		printf("freed scene\n");
#endif // DEBUG
	}

	HostScene::HostScene()
	{
		//m_Scene = new Scene();
	}

	void HostScene::initialize()
	{
		if (m_Scene == nullptr)
			m_Scene = new Scene();
	}

	HostScene::~HostScene()
	{
		printf("wouldve deleted scene\n");
		//delete m_Scene;
	}

	void HostScene::Cleanup()
	{
		if (m_Scene) {
			m_Scene->Cleanup();
			delete m_Scene;
		}
		printf("cleaned up hostscene\n");
	}

	void HostScene::addMaterial(const HostMaterial& material)
	{
		m_Scene->m_MaterialsBuffer.push_back(material.getHostMaterialData());
	}
	void HostScene::addTexture(const Texture& texture)
	{
		m_Scene->m_TexturesBuffer.push_back(texture);
	}

	uint32_t HostScene::addCamera(const HostCamera& camera)
	{
		m_Scene->m_CamerasBuffer.push_back(camera.getHostCamera());
		return m_Scene->m_CamerasBuffer.size() - 1;
	}
	void HostScene::addTriangle(const Triangle& triangle)
	{
		m_Scene->m_TrianglesBuffer.push_back(triangle);
	}
	void HostScene::addTriangleLightidx(uint32_t idx)
	{
		m_Scene->m_TriangleLightsIndicesBuffer.push_back(idx);
	}
	void HostScene::addMesh(const Mesh& mesh)
	{
		m_Scene->m_MeshesBuffer.push_back(mesh);
	}
	HostMaterial HostScene::getMaterial(uint32_t idx)
	{
		MaterialData* dmat = &m_Scene->m_MaterialsBuffer[idx];
		return HostMaterial(dmat);
	}
	Texture HostScene::getTexture(uint32_t idx)
	{
		return m_Scene->m_TexturesBuffer[idx];
	}
	const Mesh HostScene::getMesh(uint32_t idx)
	{
		return m_Scene->m_MeshesBuffer[idx];
	}
	size_t HostScene::getTrianglesBufferSize() const
	{
		return m_Scene->m_TrianglesBuffer.size();
	}
	size_t HostScene::getCamerasBufferSize() const
	{
		return m_Scene->m_CamerasBuffer.size();
	}

	size_t HostScene::getMaterialsBufferSize() const
	{
		return m_Scene->m_MaterialsBuffer.size();
	}

	size_t HostScene::getTexturesBufferSize() const
	{
		return m_Scene->m_TexturesBuffer.size();
	}

	size_t HostScene::getMeshesBufferSize() const
	{
		return m_Scene->m_MeshesBuffer.size();
	}

	size_t HostScene::getBVHNodesBufferSize() const
	{
		return m_Scene->m_BVHNodesBuffer.size();
	}

	size_t HostScene::getTriangleLightsBufferSize() const
	{
		return m_Scene->m_TriangleLightsIndicesBuffer.size();
	}
}