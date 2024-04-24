#include "RenderKernel.cuh"

#include "core/Renderer/private/Kernel/Ray.cuh"
#include "core/Renderer/private/Kernel/HitPayload.cuh"
#include "core/Renderer/private/Shapes/Scene.cuh"
#include "core/Renderer/private/Camera/Camera.cuh"
#include "core/Renderer/private/CudaMath/helper_math.cuh"//check if this requires definition activation
#include "core/Renderer/private/CudaMath/Random.cuh"

#include "Shaders/RayGen.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#define __CUDACC__ // used to get surf2d indirect functions;not how it should be done
#include <surface_indirect_functions.h>

//Render Kernel
__global__ void kernel(cudaSurfaceObject_t _surfobj, int max_x, int max_y, Camera* cam,
	const Triangle* sceneVector, size_t sceneVectorSize,
	const Material* materialvector, uint32_t frameidx, float3* accumulation_buffer)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= max_x) || (j >= max_y)) return;

	float3 fcolor = RayGen(i, j, max_x, max_y, cam, sceneVector,
		sceneVectorSize, materialvector, frameidx);

	accumulation_buffer[i + j * max_x] += fcolor;
	float3 accol = accumulation_buffer[i + j * max_x] / frameidx;
	uchar4 color = { unsigned char(255 * accol.x),unsigned char(255 * accol.y),unsigned char(255 * accol.z), 255 };

	surf2Dwrite(color, _surfobj, i * 4, j);
};

void InvokeRenderKernel(
	cudaSurfaceObject_t surfaceobj, uint32_t width, uint32_t height,
	dim3 _blocks, dim3 _threads, Camera* cam, const Scene& scene, uint32_t frameidx, float3* accumulation_buffer)
{
	const Material* DeviceMaterialVector = thrust::raw_pointer_cast(scene.m_Material.data());;
	const Triangle* DeviceSceneVector = thrust::raw_pointer_cast(scene.m_Triangles.data());
	const Mesh* DeviceMeshBuffer = thrust::raw_pointer_cast(scene.m_Meshes.data());

	kernel << < _blocks, _threads >> >
		(surfaceobj, width, height, cam, DeviceSceneVector, scene.m_Triangles.size(),
			DeviceMaterialVector, frameidx, accumulation_buffer);
}