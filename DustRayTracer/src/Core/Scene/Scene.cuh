#pragma once

#include "Core/Scene/Material.cuh"
#include "Core/Scene/Mesh.cuh"
#include "Core/Scene/RendererSettings.h"
#include "Core/Scene/Texture.cuh"

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
	const Mesh* DeviceMeshBufferPtr = nullptr;
	const BVHNode* DeviceBVHTreeRootPtr = nullptr;

	RendererSettings RenderSettings;

	size_t DeviceTextureBufferSize = 0;//unused
	size_t DeviceMaterialBufferSize = 0;//unused
	size_t DeviceMeshBufferSize = 0;
	size_t DevicePrimitivesBufferSize = 0;
};

struct Scene
{
	thrust::device_vector<Mesh> m_Meshes;
	thrust::device_vector<Material> m_Material;
	thrust::device_vector<Texture>m_Textures;
	thrust::universal_vector<Triangle>m_PrimitivesBuffer;
	BVHNode* d_BVHTreeRoot = nullptr;
	bool loadGLTFmodel(const char* filepath);

	~Scene();
private:
	bool loadMaterials(tinygltf::Model& model);
	bool loadTextures(tinygltf::Model& model, bool is_binary);
};