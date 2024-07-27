#pragma once

#include <glad/glad.h>//required by cuda_gl_interop
#include <cuda_gl_interop.h>

#include <cstdint>

struct SceneData;
namespace DustRayTracer {
	struct CameraData;
}
struct RendererSettings;//why is this here

void invokeRenderKernel(
	cudaSurfaceObject_t surfaceobj, uint32_t width, uint32_t height,
	dim3 _blocks, dim3 _threads, const DustRayTracer::CameraData* device_camera,
	const SceneData& scene_data, uint32_t frameidx, float3* accumulation_buffer);