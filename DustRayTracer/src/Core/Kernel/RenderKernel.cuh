#pragma once

#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include <cstdint>

struct Scene;
class Camera;
struct RendererSettings;

void InvokeRenderKernel(
	cudaSurfaceObject_t surfaceobj, uint32_t width, uint32_t height,
	dim3 _blocks, dim3 _threads, Camera* cam, const Scene& scene, const RendererSettings& settings, uint32_t frameidx, float3* accumulation_buffer);