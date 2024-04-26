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
	__device__ float3 getPixel(float2 UV) const;

	void Cleanup();

	int width, height = 0;
	int componentCount = 0;
private:
	unsigned char* d_data;
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
	bool loadTextures(tinygltf::Model& model);
};