#include "RenderKernel.cuh"

#include "Core/Scene/Scene.cuh"
#include "Core/Scene/SceneData.cuh"
#include "Core/PostProcess.cuh"
#include "Shaders/RayGen.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define __CUDACC__ // used to get surf2d indirect functions;not how it should be done
#include <surface_indirect_functions.h>

__global__ void integratorKernel(cudaSurfaceObject_t surface_object, int max_x, int max_y,
	const DustRayTracer::CameraData* device_camera, uint32_t frameidx, float3* accumulation_buffer, const SceneData scenedata);

//TODO: fix inconsistent buffer and primitive-triangle naming
void invokeRenderKernel(
	cudaSurfaceObject_t surfaceobj, uint32_t width, uint32_t height,
	dim3 _blocks, dim3 _threads, const DustRayTracer::CameraData* device_camera,
	const SceneData& scene_data, uint32_t frameidx, float3* accumulation_buffer)
{
	integratorKernel << < _blocks, _threads >> > (surfaceobj, width, height, device_camera, frameidx, accumulation_buffer, scene_data);
}

//Monte Carlo Render Kernel
__global__ void integratorKernel(cudaSurfaceObject_t surface_object, int max_x, int max_y, const DustRayTracer::CameraData* device_camera, 
	uint32_t frameidx, float3* accumulation_buffer, const SceneData scenedata)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= max_x) || (j >= max_y)) return;

	//raygen is the integration solver, the renderloop is integrator
	float3 sampled_radiance = rayGen(i, j, max_x, max_y, device_camera, frameidx, scenedata);//just some encapsulation; generally raygen is integrator

	//Monte carlo
	accumulation_buffer[i + j * max_x] += sampled_radiance;
	float3 estimated_radiance = accumulation_buffer[i + j * max_x] / frameidx;

	float3 processed_radiance = estimated_radiance;

	if (scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::NORMALMODE || scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::ALBEDO_DEBUG)
	{
		//order matters
		if (scenedata.RenderSettings.enable_tone_mapping)processed_radiance = toneMapping(processed_radiance, device_camera->exposure);
		if (scenedata.RenderSettings.enable_gamma_correction)processed_radiance = gammaCorrection(processed_radiance);//inverse EOTF
	}

	float4 color = { processed_radiance.x, processed_radiance.y, processed_radiance.z, 1 };

	surf2Dwrite(color, surface_object, i * (int)sizeof(float4), j);//has to be uchar4/2/1 or float4/2/1; no 3 comp color
};


