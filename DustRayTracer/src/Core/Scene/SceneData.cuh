#pragma once
#include "RendererSettings.hpp"
//#include "Camera.hpp"
//Only for internal render call consumption

namespace DustRayTracer {
	struct CameraData;
	class MaterialData;
}
class Texture;
class Triangle;
class Mesh;
class BVHNode;

//READ only; pass as const
struct SceneData
{
	SceneData() = default;
	SceneData(Texture* device_texture_buffer_ptr, size_t device_texture_buffer_size)
		:DeviceTextureBufferPtr(device_texture_buffer_ptr), DeviceTextureBufferSize(device_texture_buffer_size) {};

	const Texture* DeviceTextureBufferPtr = nullptr;
	const DustRayTracer::MaterialData* DeviceMaterialBufferPtr = nullptr;
	const Triangle* DevicePrimitivesBuffer = nullptr;
	const unsigned int* DeviceBVHPrimitiveIndicesBuffer = nullptr;
	const int* DeviceMeshLightsBufferPtr = nullptr;
	const Mesh* DeviceMeshBufferPtr = nullptr;
	DustRayTracer::CameraData* DeviceCameraBufferPtr = nullptr;
	const BVHNode* DeviceBVHNodesBuffer = nullptr;

	RendererSettings RenderSettings;

	size_t DeviceBVHNodesBufferSize = 0;
	size_t DeviceTextureBufferSize = 0;//unused
	size_t DeviceMaterialBufferSize = 0;//unused
	size_t DeviceMeshBufferSize = 0;
	size_t DevicePrimitivesBufferSize = 0;
	size_t DeviceBVHPrimitiveIndicesBufferSize = 0;
	size_t DeviceMeshLightsBufferSize = 0;
	size_t DeviceCameraBufferSize = 0;
};