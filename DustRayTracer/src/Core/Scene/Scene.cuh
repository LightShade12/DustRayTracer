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

struct Scene
{
	thrust::universal_vector<Mesh> m_Meshes;
	thrust::universal_vector<Material> m_Materials;
	thrust::device_vector<Texture>m_Textures;
	thrust::device_vector<BVHNode>m_BVHNodes;
	//lights vector
	thrust::device_vector<int>m_MeshLights;
	thrust::universal_vector<unsigned int>m_BVHTrianglesIndices;
	thrust::universal_vector<Triangle>m_PrimitivesBuffer;
	BVHNode* d_BVHTreeRoot = nullptr;//TODO: Why???
	bool loadGLTFmodel(const char* filepath, Camera** camera);

	~Scene();
private:
	bool loadMaterials(const tinygltf::Model& model);
	bool loadTextures(const tinygltf::Model& model, bool is_binary);
};