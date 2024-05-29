#include "RenderKernel.cuh"

#include "Core/Ray.cuh"
#include "Core/HitPayload.cuh"
#include "Core/BVH/BVHNode.cuh"
#include "Core/Scene/Scene.cuh"
#include "Core/Scene/Camera.cuh"
#include "Core/CudaMath/helper_math.cuh"//check if this requires definition activation
#include "Core/CudaMath/Random.cuh"

#include "Shaders/RayGen.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#define __CUDACC__ // used to get surf2d indirect functions;not how it should be done
#include <surface_indirect_functions.h>

//Render Kernel
__global__ void kernel(cudaSurfaceObject_t _surfobj, int max_x, int max_y, Camera* cam, uint32_t frameidx, float3* accumulation_buffer, const SceneData scenedata)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= max_x) || (j >= max_y)) return;

	float3 fcolor = RayGen(i, j, max_x, max_y, cam, frameidx, scenedata);

	accumulation_buffer[i + j * max_x] += fcolor;
	float3 accol = accumulation_buffer[i + j * max_x] / frameidx;
	float4 color = { accol.x, accol.y, accol.z, 1 };
	//uchar4 color = { unsigned char(255 * accol.x),unsigned char(255 * accol.y),unsigned char(255 * accol.z), 255 };

	surf2Dwrite(color, _surfobj, i * (int)sizeof(float4), j);//has to be uchar4/2/1 or float4/2/1; no 3 comp color
};

void InvokeRenderKernel(
	cudaSurfaceObject_t surfaceobj, uint32_t width, uint32_t height,
	dim3 _blocks, dim3 _threads, Camera* cam, const Scene& scene, const RendererSettings& settings, uint32_t frameidx, float3* accumulation_buffer)
{
	SceneData scenedata;
	scenedata.DeviceTextureBufferPtr = thrust::raw_pointer_cast(scene.m_Textures.data());
	scenedata.DeviceMeshBufferPtr = thrust::raw_pointer_cast(scene.m_Meshes.data());
	scenedata.DeviceMaterialBufferPtr = thrust::raw_pointer_cast(scene.m_Material.data());
	scenedata.DevicePrimitivesBuffer = thrust::raw_pointer_cast(scene.m_PrimitivesBuffer.data());
	scenedata.DeviceMeshBufferSize = scene.m_Meshes.size();
	scenedata.DevicePrimitivesBufferSize = scene.m_PrimitivesBuffer.size();
	scenedata.DeviceBVHTreeRootPtr = scene.d_BVHTreeRoot;
	scenedata.RenderSettings = settings;
	//printf("pointer: %p\n", scenedata.DeviceMeshBufferPtr);
	//printf("pointer: %p\n", scenedata.DeviceMaterialBufferPtr);
	//printf("pointer: %p\n", scenedata.DeviceTextureBufferPtr);
	//printf("pointer: %p\n", scene.d_BVHTreeRoot);

	kernel << < _blocks, _threads >> > (surfaceobj, width, height, cam, frameidx, accumulation_buffer, scenedata);
}