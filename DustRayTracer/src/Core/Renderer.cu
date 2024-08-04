#include "Core/Renderer.hpp"

#include "Core/Ray.cuh"
#include "Core/Scene/HostCamera.hpp"
#include "Core/Scene/CameraData.cuh"
#include "Core/Scene/Scene.cuh"

#include "Core/Common/CudaCommon.cuh"
#include "Kernel/RenderKernel.cuh"

#include <cuda_gl_interop.h>//for member cuda objects
#include <thrust/device_vector.h>
#include <iostream>

struct CudaAPIResource {
	CudaAPIResource() = default;
	thrust::device_vector<float3>ColorDataBuffer;
	cudaGraphicsResource_t m_RenderTargetTextureCudaResource;
	cudaEvent_t start, stop;
	int m_ThreadBlock_x = 8;
	int m_ThreadBlock_y = 8;
	dim3 m_BlockGridDimensions;
	dim3 m_ThreadBlockDimensions;
	SceneData m_DeviceSceneData;
	~CudaAPIResource()
	{
		ColorDataBuffer.clear();
	}
};

namespace DustRayTracer {
	PathTracerRenderer::PathTracerRenderer()
	{
		m_CudaAPIResource = new CudaAPIResource();
		m_CudaAPIResource->m_BlockGridDimensions = dim3(m_BufferWidth / m_CudaAPIResource->m_ThreadBlock_x + 1, m_BufferHeight / m_CudaAPIResource->m_ThreadBlock_y + 1);
		m_CudaAPIResource->m_ThreadBlockDimensions = dim3(m_CudaAPIResource->m_ThreadBlock_x, m_CudaAPIResource->m_ThreadBlock_y);
		cudaEventCreate(&(m_CudaAPIResource->start));
		cudaEventCreate(&(m_CudaAPIResource->stop));
	}

	void PathTracerRenderer::resizeResolution(uint32_t width, uint32_t height) {
		if (width == m_BufferWidth && height == m_BufferHeight)return;
		m_BufferWidth = width;
		m_BufferHeight = height;

		if (m_RenderTargetTexture_name)
		{
			m_CudaAPIResource->m_BlockGridDimensions = dim3(m_BufferWidth / m_CudaAPIResource->m_ThreadBlock_x + 1,
				m_BufferHeight / m_CudaAPIResource->m_ThreadBlock_y + 1);
			m_CudaAPIResource->m_ThreadBlockDimensions = dim3(m_CudaAPIResource->m_ThreadBlock_x, m_CudaAPIResource->m_ThreadBlock_y);

			m_CudaAPIResource->ColorDataBuffer.resize(m_BufferHeight * m_BufferWidth);

			// unregister
			cudaGraphicsUnregisterResource(m_CudaAPIResource->m_RenderTargetTextureCudaResource);
			// resize
			glBindTexture(GL_TEXTURE_2D, m_RenderTargetTexture_name);
			{
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_BufferWidth, m_BufferHeight, 0, GL_RGBA, GL_FLOAT, NULL);
			}
			glBindTexture(GL_TEXTURE_2D, 0);
			// register back
			cudaGraphicsGLRegisterImage(&(m_CudaAPIResource->m_RenderTargetTextureCudaResource), m_RenderTargetTexture_name, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
		}

		//Image recreation
		else
		{
			m_CudaAPIResource->m_BlockGridDimensions = dim3(m_BufferWidth / m_CudaAPIResource->m_ThreadBlock_x + 1,
				m_BufferHeight / m_CudaAPIResource->m_ThreadBlock_y + 1);
			m_CudaAPIResource->m_ThreadBlockDimensions = dim3(m_CudaAPIResource->m_ThreadBlock_x, m_CudaAPIResource->m_ThreadBlock_y);
			m_CudaAPIResource->ColorDataBuffer.resize(m_BufferHeight * m_BufferWidth);

			//GL texture configure
			glGenTextures(1, &m_RenderTargetTexture_name);
			glBindTexture(GL_TEXTURE_2D, m_RenderTargetTexture_name);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			//TODO: make a switchable frame filtering mode
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_BufferWidth, m_BufferHeight, 0, GL_RGBA, GL_FLOAT, NULL);

			glBindTexture(GL_TEXTURE_2D, 0);

			cudaGraphicsGLRegisterImage(&(m_CudaAPIResource->m_RenderTargetTextureCudaResource), m_RenderTargetTexture_name, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
		}

		clearAccumulation();
	}

	void PathTracerRenderer::renderFrame(float* delta)
	{
		if (m_FrameIndex == m_RendererSettings.max_samples)return;

		cudaGraphicsMapResources(1, &(m_CudaAPIResource->m_RenderTargetTextureCudaResource));

		cudaArray_t render_target_texture_sub_resource_array;
		cudaGraphicsSubResourceGetMappedArray(&render_target_texture_sub_resource_array, (m_CudaAPIResource->m_RenderTargetTextureCudaResource), 0, 0);
		cudaResourceDesc render_target_texture_resource_descriptor;
		{
			render_target_texture_resource_descriptor.resType = cudaResourceTypeArray;
			render_target_texture_resource_descriptor.res.array.array = render_target_texture_sub_resource_array;
		}
		cudaSurfaceObject_t render_target_texture_surface_object;
		cudaCreateSurfaceObject(&render_target_texture_surface_object, &render_target_texture_resource_descriptor);

		//----
		cudaEventRecord(m_CudaAPIResource->start);
		invokeRenderKernel(render_target_texture_surface_object, m_BufferWidth, m_BufferHeight,
			m_CudaAPIResource->m_BlockGridDimensions, m_CudaAPIResource->m_ThreadBlockDimensions, m_CurrentCamera.getDeviceCamera(), m_CudaAPIResource->m_DeviceSceneData, m_FrameIndex,
			thrust::raw_pointer_cast(m_CudaAPIResource->ColorDataBuffer.data()));

		checkCudaErrors(cudaGetLastError());

		cudaEventRecord(m_CudaAPIResource->stop);
		checkCudaErrors(cudaEventSynchronize(m_CudaAPIResource->stop));

		cudaEventElapsedTime(delta, m_CudaAPIResource->start, m_CudaAPIResource->stop);
		checkCudaErrors(cudaDeviceSynchronize());
		//----

			//post render cuda---------------------------------------------------------------------------------
		cudaDestroySurfaceObject(render_target_texture_surface_object);
		cudaGraphicsUnmapResources(1, &(m_CudaAPIResource->m_RenderTargetTextureCudaResource));
		cudaStreamSynchronize(0);
		m_FrameIndex++;
	}

	PathTracerRenderer::~PathTracerRenderer()
	{
		//DRT
		m_CurrentCamera.cleanup();
		//CUDA
		cudaEventDestroy(m_CudaAPIResource->start);
		cudaEventDestroy(m_CudaAPIResource->stop);
		delete m_CudaAPIResource;
		//GL
		glDeleteTextures(1, &m_RenderTargetTexture_name);
	}

	bool PathTracerRenderer::initialize()
	{
#ifdef DEBUG
		printf("Renderer initialized\n");
#endif // DEBUG

		return true;
	}

	bool PathTracerRenderer::shutdown()
	{
#ifdef DEBUG
		printf("Renderer shutdown sucessfully\n");
#endif // DEBUG
		return true;
	}

	void PathTracerRenderer::updateScene(DustRayTracer::HostScene& scene_object)
	{
		m_CudaAPIResource->m_DeviceSceneData.DeviceBVHNodesBuffer = thrust::raw_pointer_cast(scene_object.m_Scene->m_BVHNodesBuffer.data());
		m_CudaAPIResource->m_DeviceSceneData.DeviceBVHPrimitiveIndicesBuffer = thrust::raw_pointer_cast(scene_object.m_Scene->m_BVHTrianglesIndicesBuffer.data());
		m_CudaAPIResource->m_DeviceSceneData.DevicePrimitivesBuffer = thrust::raw_pointer_cast(scene_object.m_Scene->m_TrianglesBuffer.data());
		m_CudaAPIResource->m_DeviceSceneData.DeviceTextureBufferPtr = thrust::raw_pointer_cast(scene_object.m_Scene->m_TexturesBuffer.data());
		m_CudaAPIResource->m_DeviceSceneData.DeviceMaterialBufferPtr = thrust::raw_pointer_cast(scene_object.m_Scene->m_MaterialsBuffer.data());
		m_CudaAPIResource->m_DeviceSceneData.DeviceMeshBufferPtr = thrust::raw_pointer_cast(scene_object.m_Scene->m_MeshesBuffer.data());
		m_CudaAPIResource->m_DeviceSceneData.DeviceCameraBufferPtr = thrust::raw_pointer_cast(scene_object.m_Scene->m_CamerasBuffer.data());
		m_CudaAPIResource->m_DeviceSceneData.DeviceMeshLightsBufferPtr = thrust::raw_pointer_cast(scene_object.m_Scene->m_TriangleLightsIndicesBuffer.data());
		//----
		m_CudaAPIResource->m_DeviceSceneData.DeviceMeshBufferSize = scene_object.m_Scene->m_MeshesBuffer.size();
		m_CudaAPIResource->m_DeviceSceneData.DevicePrimitivesBufferSize = scene_object.m_Scene->m_TrianglesBuffer.size();
		m_CudaAPIResource->m_DeviceSceneData.DeviceBVHPrimitiveIndicesBufferSize = scene_object.m_Scene->m_BVHTrianglesIndicesBuffer.size();
		m_CudaAPIResource->m_DeviceSceneData.DeviceMeshLightsBufferSize = scene_object.m_Scene->m_TriangleLightsIndicesBuffer.size();
		m_CudaAPIResource->m_DeviceSceneData.DeviceBVHNodesBufferSize = scene_object.m_Scene->m_BVHNodesBuffer.size();
		m_CudaAPIResource->m_DeviceSceneData.DeviceCameraBufferSize = scene_object.m_Scene->m_CamerasBuffer.size();
		//---
		DustRayTracer::CameraData* cam = &scene_object.m_Scene->m_CamerasBuffer[0];
		m_CurrentCamera = HostCamera(thrust::raw_pointer_cast(cam));
		m_CudaAPIResource->m_DeviceSceneData.RenderSettings = m_RendererSettings;
	}

	void PathTracerRenderer::updateRendererConfig(const RendererSettings& config)
	{
		{ m_RendererSettings = config; m_CudaAPIResource->m_DeviceSceneData.RenderSettings = m_RendererSettings; }
	}

	/*HostCamera PathTracerRenderer::getCamera() const
	{
		return m_CurrentCamera;
	}*/

	DustRayTracer::HostCamera* PathTracerRenderer::getCameraPtr()
	{
		return &m_CurrentCamera;
	}

	void PathTracerRenderer::changeCamera(uint32_t camera_idx)
	{
		if (camera_idx >= m_CudaAPIResource->m_DeviceSceneData.DeviceCameraBufferSize)camera_idx = m_CudaAPIResource->m_DeviceSceneData.DeviceCameraBufferSize - 1;
		else if (camera_idx < 0)camera_idx = 0;
		DustRayTracer::CameraData* cm = &m_CudaAPIResource->m_DeviceSceneData.DeviceCameraBufferPtr[camera_idx];
		m_CurrentCamera = HostCamera(thrust::raw_pointer_cast(cm));
	}

	void PathTracerRenderer::clearAccumulation()
	{
		thrust::fill(m_CudaAPIResource->ColorDataBuffer.begin(), m_CudaAPIResource->ColorDataBuffer.end(), make_float3(0, 0, 0));
		m_FrameIndex = 1;
	}
}