#pragma once
#include "Core/Public/DustRayTracer.hpp"

namespace tinygltf {
	class Model;
}

class Importer
{
public:
	Importer() = default;
	bool loadGLTF(const char* filepath, DustRayTracer::HostScene& scene_object);

private:
	DustRayTracer::HostScene* m_WorkingScene = nullptr;
	bool loadMaterials(const tinygltf::Model& model);
	bool loadTextures(const tinygltf::Model& model, bool is_binary);
};
