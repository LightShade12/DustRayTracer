#include "core/Renderer/Renderer.hpp"
#include "core/Renderer/private/Camera/Camera.cuh"

#include "core/Common/CudaCommon.cuh"
#include "Kernel/RenderKernel.cuh"
#include <thrust/device_vector.h>
#include <iostream>

struct FrameBufferWrapper
{
	thrust::device_vector<float3>ColorDataBuffer;
};

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

Renderer::Renderer()
{
	m_AccumulationBuffer = new FrameBufferWrapper();

	blocks = dim3(m_BufferWidth / tx + 1, m_BufferHeight / ty + 1);
	threads = dim3(tx, ty);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

void Renderer::ResizeBuffer(uint32_t width, uint32_t height) {
	if (width == m_BufferWidth && height == m_BufferHeight)return;
	m_BufferWidth = width;
	m_BufferHeight = height;

	//m_AccumulationBuffer->ColorDataBuffer.clear();

	if (m_RenderTarget_name)
	{
		blocks = dim3(m_BufferWidth / tx + 1, m_BufferHeight / ty + 1);
		threads = dim3(tx, ty);

		m_AccumulationBuffer->ColorDataBuffer.resize(m_BufferHeight * m_BufferWidth);

		// unregister
		cudaGraphicsUnregisterResource(m_viewCudaResource);
		// resize
		glBindTexture(GL_TEXTURE_2D, m_RenderTarget_name);
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_BufferWidth, m_BufferHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		}
		glBindTexture(GL_TEXTURE_2D, 0);
		// register back
		cudaGraphicsGLRegisterImage(&m_viewCudaResource, m_RenderTarget_name, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	}

	//Image recreation
	else
	{
		blocks = dim3(m_BufferWidth / tx + 1, m_BufferHeight / ty + 1);
		threads = dim3(tx, ty);
		m_AccumulationBuffer->ColorDataBuffer.resize(m_BufferHeight * m_BufferWidth);
		//GL texture configure
		glGenTextures(1, &m_RenderTarget_name);
		glBindTexture(GL_TEXTURE_2D, m_RenderTarget_name);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_BufferWidth, m_BufferHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glBindTexture(GL_TEXTURE_2D, 0);

		cudaGraphicsGLRegisterImage(&m_viewCudaResource, m_RenderTarget_name, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	}

	resetAccumulationBuffer();
}

void Renderer::Render(Camera* cam, const Scene& scene, float* delta)
{
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

	//printf("acc buffer size: %d", m_AccumulationBuffer->ColorDataBuffer.size());
	InvokeRenderKernel(viewCudaSurfaceObject, m_BufferWidth, m_BufferHeight,
		blocks, threads, cam, scene, m_FrameIndex,
		thrust::raw_pointer_cast(m_AccumulationBuffer->ColorDataBuffer.data()));
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
	return m_RenderTarget_name;
}

Renderer::~Renderer()
{
	delete m_AccumulationBuffer;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	glDeleteTextures(1, &m_RenderTarget_name);
}

void Renderer::resetAccumulationBuffer()
{
	thrust::fill(m_AccumulationBuffer->ColorDataBuffer.begin(), m_AccumulationBuffer->ColorDataBuffer.end(), float3());
	m_FrameIndex = 1;
}
