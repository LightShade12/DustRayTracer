#include "core/Renderer/Renderer.hpp"
#include "core/Renderer/private/Camera/Camera.cuh"

#include "core/Editor/Common/CudaCommon.cuh"
#include "Kernel/RenderKernel.cuh"
#include <thrust/device_vector.h>
#include <iostream>

struct ThrustRGB32FBufferWrapper
{
	thrust::device_vector<float3>ColorDataBuffer;

	~ThrustRGB32FBufferWrapper() 
	{
		ColorDataBuffer.clear();
	}
};

Renderer::Renderer()
{
	m_AccumulationFrameBuffer = new ThrustRGB32FBufferWrapper();

	blocks = dim3(m_BufferWidth / tx + 1, m_BufferHeight / ty + 1);
	threads = dim3(tx, ty);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

void Renderer::ResizeBuffer(uint32_t width, uint32_t height) {
	if (width == m_BufferWidth && height == m_BufferHeight)return;
	m_BufferWidth = width;
	m_BufferHeight = height;

	//m_AccumulationFrameBuffer->ColorDataBuffer.clear();

	if (m_RenderTargetTexture_name)
	{
		blocks = dim3(m_BufferWidth / tx + 1, m_BufferHeight / ty + 1);
		threads = dim3(tx, ty);

		m_AccumulationFrameBuffer->ColorDataBuffer.resize(m_BufferHeight * m_BufferWidth);

		// unregister
		cudaGraphicsUnregisterResource(m_viewCudaResource);
		// resize
		glBindTexture(GL_TEXTURE_2D, m_RenderTargetTexture_name);
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_BufferWidth, m_BufferHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		}
		glBindTexture(GL_TEXTURE_2D, 0);
		// register back
		cudaGraphicsGLRegisterImage(&m_viewCudaResource, m_RenderTargetTexture_name, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	}

	//Image recreation
	else
	{
		blocks = dim3(m_BufferWidth / tx + 1, m_BufferHeight / ty + 1);
		threads = dim3(tx, ty);
		m_AccumulationFrameBuffer->ColorDataBuffer.resize(m_BufferHeight * m_BufferWidth);
		//GL texture configure
		glGenTextures(1, &m_RenderTargetTexture_name);
		glBindTexture(GL_TEXTURE_2D, m_RenderTargetTexture_name);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_BufferWidth, m_BufferHeight, 0, GL_RGBA, GL_FLOAT, NULL);

		glBindTexture(GL_TEXTURE_2D, 0);

		cudaGraphicsGLRegisterImage(&m_viewCudaResource, m_RenderTargetTexture_name, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	}

	resetAccumulationBuffer();
}

void Renderer::Render(Camera* cam, const Scene& scene, float* delta)
{
	if (m_FrameIndex == m_RendererSettings.max_samples)return;

	cudaGraphicsMapResources(1, &m_viewCudaResource);

	cudaArray_t viewCudaArray;
	cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, m_viewCudaResource, 0, 0);
	cudaResourceDesc viewCudaArrayResourceDesc;
	{
		viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
		viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
	}
	cudaSurfaceObject_t viewCudaSurfaceObject;
	cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc);

	//----
	cudaEventRecord(start);

	//printf("acc buffer size: %d", m_AccumulationFrameBuffer->ColorDataBuffer.size());
	InvokeRenderKernel(viewCudaSurfaceObject, m_BufferWidth, m_BufferHeight,
		blocks, threads, cam, scene, m_RendererSettings, m_FrameIndex,
		thrust::raw_pointer_cast(m_AccumulationFrameBuffer->ColorDataBuffer.data()));
	checkCudaErrors(cudaGetLastError());

	cudaEventRecord(stop);
	checkCudaErrors(cudaEventSynchronize(stop));

	cudaEventElapsedTime(delta, start, stop);
	//checkCudaErrors(cudaDeviceSynchronize());
	//----

	//post render cuda---------------------------------------------------------------------------------
	cudaDestroySurfaceObject(viewCudaSurfaceObject);
	cudaGraphicsUnmapResources(1, &m_viewCudaResource);
	cudaStreamSynchronize(0);
	m_FrameIndex++;
}

GLuint& Renderer::GetRenderTargetImage_name()
{
	return m_RenderTargetTexture_name;
}

Renderer::~Renderer()
{
	delete m_AccumulationFrameBuffer;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	glDeleteTextures(1, &m_RenderTargetTexture_name);
}

void Renderer::resetAccumulationBuffer()
{
	thrust::fill(m_AccumulationFrameBuffer->ColorDataBuffer.begin(), m_AccumulationFrameBuffer->ColorDataBuffer.end(), float3());
	m_FrameIndex = 1;
}