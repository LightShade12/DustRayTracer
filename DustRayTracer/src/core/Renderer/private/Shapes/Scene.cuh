#pragma once

#include "core/Renderer/private/Shapes/Material.cuh"
#include "core/Renderer/private/Shapes/Mesh.cuh"

#include <thrust/device_vector.h>


namespace tinygltf
{
	class Model;
}

struct Scene
{
	thrust::device_vector<Mesh> m_Meshes;
	thrust::device_vector<Material> m_Material;

	bool loadGLTFmodel(const char* filepath);

	~Scene();
private:
	bool loadMaterials(tinygltf::Model& model);

};