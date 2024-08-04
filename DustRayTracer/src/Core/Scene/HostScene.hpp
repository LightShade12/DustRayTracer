#pragma once
/*
To be owned by client.
Provides safe methods to modify contents from host side and sync with device
*/
#include "Core/Scene/Mesh.cuh"
#include "Core/Scene/Material.cuh"
#include "Core/Scene/Texture.hpp"
#include "Core/Scene/HostCamera.hpp"
#include "Core/Scene/Triangle.cuh"
//#include "Core/BVH/BVHNode.cuh"

namespace DustRayTracer {
	struct Scene;

	class HostScene {
	public:
		HostScene();
		void initialize();

		~HostScene();
		void Cleanup();

		void addMaterial(const HostMaterial& material);
		void addTexture(const Texture& texture);
		uint32_t addCamera(const HostCamera& camera);//returns index of the added camera
		void addTriangle(const Triangle& triangle);
		void addTriangleLightidx(uint32_t idx);
		void addMesh(const Mesh& mesh);

		HostMaterial getMaterial(uint32_t idx);
		Texture getTexture(uint32_t idx);
		const Mesh getMesh(uint32_t idx);//make this non const and impl hostmesh if you wanna modify it in editor

		size_t getTrianglesBufferSize() const;
		size_t getCamerasBufferSize() const;
		size_t getMaterialsBufferSize() const;
		size_t getTexturesBufferSize() const;
		size_t getMeshesBufferSize() const;
		size_t getBVHNodesBufferSize() const;
		size_t getTriangleLightsBufferSize() const;

		Scene* m_Scene = nullptr;
	};
}