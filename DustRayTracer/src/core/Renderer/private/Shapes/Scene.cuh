#pragma once

#include "core/Renderer/private/Shapes/Material.cuh"
#include "core/Renderer/private/Shapes/Mesh.cuh"

#include <thrust/device_vector.h>

namespace tinygltf
{
	class Model;
}

class Texture
{
public:
	Texture() = default;
	Texture(const char* filepath);
	Texture(unsigned char* data, size_t bytesize);
	__device__ float3 getPixel(float2 UV) const;
	__device__ float getAlpha(float2 UV) const;
	void Cleanup();

	int width, height = 0;
	int componentCount = 0;
private:
	unsigned char* d_data;
};

//READ only; pass as const
struct SceneData
{
	SceneData() = default;
	SceneData(Texture* device_texture_buffer_ptr, size_t device_texture_buffer_size)
		:DeviceTextureBufferPtr(device_texture_buffer_ptr), DeviceTextureBufferSize(device_texture_buffer_size) {};

	const Texture* DeviceTextureBufferPtr;
	const Material* DeviceMaterialBufferPtr;
	const Mesh* DeviceMeshBufferPtr;

	size_t DeviceTextureBufferSize = 0;//unused
	size_t DeviceMaterialBufferSize = 0;//unused
	size_t DeviceMeshBufferSize = 0;
};

struct Scene
{
	thrust::device_vector<Mesh> m_Meshes;
	thrust::device_vector<Material> m_Material;
	thrust::device_vector<Texture>m_Textures;

	bool loadGLTFmodel(const char* filepath);

	~Scene();
private:
	bool loadMaterials(tinygltf::Model& model);
	bool loadTextures(tinygltf::Model& model, bool is_binary);
};