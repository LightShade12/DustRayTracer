#pragma once
//#include "private/Camera/Camera.cuh"
#include "private/Kernel/RendererSettings.h"

#include <glad/glad.h>
#include <cuda_gl_interop.h>//for member cuda objects

#include <cstdint>

struct Scene;
class Camera;
struct FrameBufferWrapper;

class Renderer
{
public:

	Renderer();
	void ResizeBuffer(uint32_t width, uint32_t height);
	void Render(Camera* cam, const Scene& scene, float* delta);
	GLuint& GetRenderTargetImage_name();
	~Renderer();

	uint32_t getBufferWidth() const { return m_BufferWidth; }
	uint32_t getBufferHeight() const { return m_BufferHeight; }
	uint32_t getSampleCount() const { return m_FrameIndex; }
	void resetAccumulationBuffer();

public:
	RendererSettings m_RendererSettings;

private:
	uint32_t m_BufferWidth = 0, m_BufferHeight = 0;
	GLuint m_RenderTarget_name = NULL;//null init val important for triggering init

	cudaGraphicsResource_t m_viewCudaResource;
	cudaEvent_t start, stop;

	FrameBufferWrapper* m_AccumulationBuffer;

	uint32_t m_FrameIndex = 1;

	int ty = 8;
	int tx = 8;
	dim3 blocks;
	dim3 threads;
};