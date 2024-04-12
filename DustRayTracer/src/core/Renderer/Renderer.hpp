#pragma once
#include "private/camera/Camera.hpp"

#include <glad/glad.h>
#include <cuda_gl_interop.h>//for member cuda objects

#include <cstdint>

struct Scene;

class Renderer
{
public:
	Renderer();
	void ResizeBuffer(uint32_t width, uint32_t height);
	void Render(Camera* cam, const Scene& scene, float* delta);
	GLuint& GetRenderTargetImage_name();
	~Renderer();

	uint32_t m_BufferWidth = 0, m_BufferHeight = 0;
private:
	GLuint m_RenderTarget_name = NULL;//null init val important for triggering init

	cudaGraphicsResource_t m_viewCudaResource;
	cudaEvent_t start, stop;

	int ty = 8;
	int tx = 8;
	dim3 blocks;
	dim3 threads;
};