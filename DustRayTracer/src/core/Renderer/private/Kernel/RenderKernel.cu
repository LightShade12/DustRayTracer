#include "RenderKernel.cuh"

#include "core/Renderer/private/Kernel/Ray.cuh"
#include "core/Renderer/private/Kernel/HitPayload.cuh"
#include "core/Renderer/private/Shapes/Scene.cuh"
#include "core/Renderer/private/Camera/Camera.cuh"
#include "core/Renderer/private/CudaMath/helper_math.cuh"//check if this requires definition activation
#include "core/Renderer/private/CudaMath/Random.cuh"

#include "Shaders/ClosestHit.cuh"
#include "Shaders/Miss.cuh"
#include "Shaders/Intersection.cuh"
#include "Shaders/RayGen.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#define __CUDACC__ // used to get surf2d indirect functions;not how it should be done
#include <surface_indirect_functions.h>

//traverse accel struct
__device__ HitPayload TraceRay(const Ray& ray, const Sphere* scene_vector, size_t scene_vector_size) {
	int closestObjectIdx = -1;
	float hitDistance = FLT_MAX;
	HitPayload workingPayload;

	for (int i = 0; i < scene_vector_size; i++)
	{
		const Sphere* sphere = &scene_vector[i];
		workingPayload = Intersection(ray, sphere);

		if (workingPayload.hit_distance < hitDistance && workingPayload.hit_distance>0)
		{
			hitDistance = workingPayload.hit_distance;
			closestObjectIdx = i;
		}
	}

	if (closestObjectIdx < 0)
	{
		return Miss(ray);
	}

	return ClosestHit(ray, closestObjectIdx, hitDistance, scene_vector);
};

//Render Kernel
__global__ void kernel(cudaSurfaceObject_t _surfobj, int max_x, int max_y, Camera* cam,
	const Sphere* sceneVector, size_t sceneVectorSize,
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
	//uchar4 color = { unsigned char(255 * fcolor.x),unsigned char(255 * fcolor.y),unsigned char(255 * fcolor.z), 255 };

	surf2Dwrite(color, _surfobj, i * 4, j);
};

void InvokeRenderKernel(
	cudaSurfaceObject_t surfaceobj, uint32_t width, uint32_t height,
	dim3 _blocks, dim3 _threads, Camera* cam, const Scene& scene, uint32_t frameidx, float3* accumulation_buffer)
{
	const Material* DeviceMaterialVector = thrust::raw_pointer_cast(scene.m_Material.data());;
	const Sphere* DeviceSceneVector = thrust::raw_pointer_cast(scene.m_Spheres.data());
	kernel << < _blocks, _threads >> > (surfaceobj, width, height, cam, DeviceSceneVector,
		scene.m_Spheres.size(), DeviceMaterialVector, frameidx, accumulation_buffer);
}