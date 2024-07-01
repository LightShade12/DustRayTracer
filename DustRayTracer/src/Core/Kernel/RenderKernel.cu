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

__device__ static float3 uncharted2_tonemap_partial(float3 x)
{
	float A = 0.15f;
	float B = 0.50f;
	float C = 0.10f;
	float D = 0.20f;
	float E = 0.02f;
	float F = 0.30f;
	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

__device__ static float3 uncharted2_filmic(float3 v, float exposure)
{
	float exposure_bias = exposure;
	float3 curr = uncharted2_tonemap_partial(v * exposure_bias);

	float3 W = make_float3(11.2f);
	float3 white_scale = make_float3(1.0f) / uncharted2_tonemap_partial(W);
	return curr * white_scale;
}

__device__ static float3 toneMapping(float3 HDR_color, float exposure = 2.f) {
	float3 LDR_color = uncharted2_filmic(HDR_color, exposure);
	return LDR_color;
}

__device__ static float3 gammaCorrection(const float3 linear_color) {
	float3 gamma_space_color = { sqrtf(linear_color.x),sqrtf(linear_color.y) ,sqrtf(linear_color.z) };
	return gamma_space_color;
}

//Render Kernel
__global__ void kernel(cudaSurfaceObject_t _surfobj, int max_x, int max_y, Camera* cam, uint32_t frameidx, float3* accumulation_buffer, const SceneData scenedata)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= max_x) || (j >= max_y)) return;

	//raygen is the integrator
	float3 fcolor = rayGen(i, j, max_x, max_y, cam, frameidx, scenedata);

	accumulation_buffer[i + j * max_x] += fcolor;
	float3 accol = accumulation_buffer[i + j * max_x] / frameidx;
	//post processing
	if (scenedata.RenderSettings.RenderMode == RendererSettings::RenderModes::NORMALMODE || scenedata.RenderSettings.DebugMode == RendererSettings::DebugModes::ALBEDO_DEBUG)
	{
		if (scenedata.RenderSettings.tone_mapping)accol = toneMapping(accol, cam->exposure);
		if (scenedata.RenderSettings.gamma_correction)accol = gammaCorrection(accol);
	}
	float4 color = { accol.x, accol.y, accol.z, 1 };

	surf2Dwrite(color, _surfobj, i * (int)sizeof(float4), j);//has to be uchar4/2/1 or float4/2/1; no 3 comp color
};

void InvokeRenderKernel(
	cudaSurfaceObject_t surfaceobj, uint32_t width, uint32_t height,
	dim3 _blocks, dim3 _threads, Camera* cam, const Scene& scene, const RendererSettings& settings, uint32_t frameidx, float3* accumulation_buffer)
{
	SceneData scenedata;
	scenedata.DeviceBVHNodesBuffer = thrust::raw_pointer_cast(scene.m_BVHNodes.data());
	scenedata.DeviceTextureBufferPtr = thrust::raw_pointer_cast(scene.m_Textures.data());
	scenedata.DeviceMeshBufferPtr = thrust::raw_pointer_cast(scene.m_Meshes.data());
	scenedata.DeviceMaterialBufferPtr = thrust::raw_pointer_cast(scene.m_Materials.data());
	scenedata.DevicePrimitivesBuffer = thrust::raw_pointer_cast(scene.m_PrimitivesBuffer.data());
	scenedata.DeviceMeshLightsBufferPtr = thrust::raw_pointer_cast(scene.m_MeshLights.data());
	//----
	scenedata.DeviceMeshBufferSize = scene.m_Meshes.size();
	scenedata.DevicePrimitivesBufferSize = scene.m_PrimitivesBuffer.size();
	scenedata.DeviceMeshLightsBufferSize= scene.m_MeshLights.size();
	scenedata.DeviceBVHNodesBufferSize = scene.m_BVHNodes.size();
	scenedata.DeviceBVHTreeRootPtr = scene.d_BVHTreeRoot;
	scenedata.RenderSettings = settings;

	kernel << < _blocks, _threads >> > (surfaceobj, width, height, cam, frameidx, accumulation_buffer, scenedata);
}