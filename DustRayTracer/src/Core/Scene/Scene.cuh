#pragma once

#include "Core/Scene/Material.cuh"
#include "Core/Scene/Mesh.cuh"
#include "Core/Scene/RendererSettings.h"
#include "Core/Scene/Texture.cuh"
#include  "Camera.cuh"

#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

namespace tinygltf
{
	class Model;
}

class BVHNode;

//READ only; pass as const
struct SceneData
{
	SceneData() = default;
	SceneData(Texture* device_texture_buffer_ptr, size_t device_texture_buffer_size)
		:DeviceTextureBufferPtr(device_texture_buffer_ptr), DeviceTextureBufferSize(device_texture_buffer_size) {};

	const Texture* DeviceTextureBufferPtr = nullptr;
	const Material* DeviceMaterialBufferPtr = nullptr;
	const Triangle* DevicePrimitivesBuffer = nullptr;
	const int* DeviceMeshLightsBufferPtr = nullptr;
	const Mesh* DeviceMeshBufferPtr = nullptr;
	const BVHNode* DeviceBVHTreeRootPtr = nullptr;
	const BVHNode* DeviceBVHNodesBuffer = nullptr;

	RendererSettings RenderSettings;

	size_t DeviceBVHNodesBufferSize = 0;
	size_t DeviceTextureBufferSize = 0;//unused
	size_t DeviceMaterialBufferSize = 0;//unused
	size_t DeviceMeshBufferSize = 0;
	size_t DevicePrimitivesBufferSize = 0;
	size_t DeviceMeshLightsBufferSize = 0;
};

struct Scene
{
	thrust::universal_vector<Mesh> m_Meshes;
	thrust::universal_vector<Material> m_Materials;
	thrust::device_vector<Texture>m_Textures;
	thrust::device_vector<BVHNode>m_BVHNodes;
	thrust::device_vector<int>m_MeshLights;
	//lights vector
	thrust::universal_vector<Triangle>m_PrimitivesBuffer;
	BVHNode* d_BVHTreeRoot = nullptr;//Why???
	bool loadGLTFmodel(const char* filepath, Camera** camera);

	~Scene();
private:
	bool loadMaterials(const tinygltf::Model& model);
	bool loadTextures(const tinygltf::Model& model, bool is_binary);
};