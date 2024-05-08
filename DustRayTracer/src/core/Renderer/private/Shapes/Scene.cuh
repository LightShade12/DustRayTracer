#pragma once

#include "core/Renderer/private/Shapes/Material.cuh"
#include "core/Renderer/private/Shapes/Mesh.cuh"
#include "core/Renderer/private/Kernel/RendererSettings.h"
#include "core/Renderer/private/Kernel/Texture.cuh"

#include <thrust/device_vector.h>

namespace tinygltf
{
	class Model;
}

class Node;

//READ only; pass as const
struct SceneData
{
	SceneData() = default;
	SceneData(Texture* device_texture_buffer_ptr, size_t device_texture_buffer_size)
		:DeviceTextureBufferPtr(device_texture_buffer_ptr), DeviceTextureBufferSize(device_texture_buffer_size) {};

	const Texture* DeviceTextureBufferPtr = nullptr;
	const Material* DeviceMaterialBufferPtr = nullptr;
	const Mesh* DeviceMeshBufferPtr = nullptr;
	const Node* DeviceBVHTreePtr = nullptr;

	RendererSettings RenderSettings;

	size_t DeviceTextureBufferSize = 0;//unused
	size_t DeviceMaterialBufferSize = 0;//unused
	size_t DeviceMeshBufferSize = 0;
};

struct Scene
{
	thrust::device_vector<Mesh> m_Meshes;
	thrust::device_vector<Material> m_Material;
	thrust::device_vector<Texture>m_Textures;
	thrust::device_vector<Triangle>m_PrimitivesBuffer;
	Node* d_BVHTreeRoot = nullptr;
	bool loadGLTFmodel(const char* filepath);

	~Scene();
private:
	bool loadMaterials(tinygltf::Model& model);
	bool loadTextures(tinygltf::Model& model, bool is_binary);
};