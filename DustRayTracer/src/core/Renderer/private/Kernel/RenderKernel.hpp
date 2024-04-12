#pragma once
#include <cstdint>
//#include <cuda_gl_interop.h>

struct Scene;
class Camera;

void InvokeRenderKernel(
	cudaSurfaceObject_t surfaceobj, uint32_t width, uint32_t height,
	dim3 _blocks, dim3 _threads, Camera* cam, const Scene& scene);