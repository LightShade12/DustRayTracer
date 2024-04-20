#include "RenderKernel.cuh"

#include "core/Renderer/private/Kernel/Ray.cuh"
#include "core/Renderer/private/Kernel/HitPayload.cuh"
#include "core/Renderer/private/Shapes/Scene.cuh"
#include "core/Renderer/private/Camera/Camera.cuh"
#include "core/Renderer/private/CudaMath/helper_math.cuh"//check if this requires definition activation

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#define __CUDACC__ // used to get surf2d indirect functions;not how it should be done
#include <surface_indirect_functions.h>

__device__ uint pcg_hash(uint input)
{
	uint state = input * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

//0 to 1
__device__ float randomFloat(uint32_t& seed)
{
	seed = pcg_hash(seed);
	return (float)seed / (float)UINT32_MAX;
}

//-1 to 1
__device__ float3 randomUnitVec3(uint32_t& seed)
{
	return normalize(make_float3(
		randomFloat(seed) * 2.f - 1.f,
		randomFloat(seed) * 2.f - 1.f,
		randomFloat(seed) * 2.f - 1.f));
}

__device__ float3 randomUnitSphereVec3(uint32_t& seed) {
	while (true) {
		float3 p = randomUnitVec3(seed);
		float len = length(p);
		if ((len * len) < 1)
			return p;
	}
}

__device__ HitPayload ClosestHit(const Ray& ray, uint32_t obj_idx, float hit_distance, const Sphere* scene_vec) {
	const Sphere* sphere = &(scene_vec[obj_idx]);

	float3 origin = ray.origin - sphere->Position;//apply sphere translation

	HitPayload payload;
	payload.hit_distance = hit_distance;
	payload.world_position = origin + ray.direction * hit_distance;//hit position
	payload.world_normal = normalize(payload.world_position);
	payload.object_idx = obj_idx;

	payload.world_position += sphere->Position;

	return payload;
};

__device__ HitPayload Miss(const Ray& ray) {
	HitPayload payload;
	payload.hit_distance = -1;
	return payload;
};

__device__ HitPayload TraceRay(const Ray& ray, const Sphere* scene_vector, size_t scene_vector_size) {
	int closestObjectIdx = -1;
	float hitDistance = FLT_MAX;

	for (int i = 0; i < scene_vector_size; i++)
	{
		const Sphere* sphere = &scene_vector[i];
		float3 origin = ray.origin - sphere->Position;

		float a = dot(ray.direction, ray.direction);
		float b = 2.0f * dot(origin, ray.direction);
		float c = dot(origin, origin) - sphere->Radius * sphere->Radius;

		float discriminant = b * b - 4.0f * a * c;
		if (discriminant < 0.0f)
			continue;

		float closestT = (-b - sqrt(discriminant)) / (2.0f * a);
		if (closestT < hitDistance && closestT>0)
		{
			hitDistance = closestT;
			closestObjectIdx = i;
		}
	}

	if (closestObjectIdx < 0)
	{
		return Miss(ray);
	}

	return ClosestHit(ray, closestObjectIdx, hitDistance, scene_vector);
};

__device__ float3 RayGen(uint32_t x, uint32_t y, uint32_t max_x, uint32_t max_y,
	const Camera* cam, const Sphere* scene_vector, size_t scenevecsize, const Material* matvector, uint32_t frameidx) {
	float2 uv = { (float(x) / max_x) ,(float(y) / max_y) };

	//uv.x *= ((float)max_x / (float)max_y);
	//uv.x = uv.x * 2.f - ((float)max_x / (float)max_y);
	//uv.y = uv.y * 2.f - 1.f;
	uv = uv * 2 - 1;

	Ray ray;
	ray.origin = cam->m_Position;
	ray.direction = cam->GetRayDir(uv, 30, max_x, max_y);
	float3 color = { 0,0,0 };

	uint32_t seed = x + y * max_x;
	seed *= frameidx;

	float multiplier = 1.f;
	int bounces = 10;
	for (int i = 0; i < bounces; i++)
	{
		HitPayload payload = TraceRay(ray, scene_vector, scenevecsize);
		seed += i;
		//sky
		if (payload.hit_distance < 0)
		{
			float a = 0.5 * (1 + (normalize(ray.direction)).y);
			float3 col1 = { 0.5,0.7,1.0 };
			float3 col2 = { 1,1,1 };
			float3 fcol = (float(1 - a) * col2) + (a * col1);
			color += fcol * multiplier;
			break;
		}

		float3 lightDir = normalize(make_float3(-1, -1, -1));
		float lightIntensity = max(dot(payload.world_normal, -lightDir), 0.0f); // == cos(angle)

		const Sphere closestsphere = scene_vector[payload.object_idx];
		float3 spherecolor = matvector[closestsphere.MaterialIndex].Albedo;
		//spherecolor *= lightIntensity;
		//color += spherecolor * multiplier;

		multiplier *= 0.5f;

		ray.origin = payload.world_position + (payload.world_normal * 0.0001f);
		//ray.direction = reflect(ray.direction, payload.world_normal + (normalize(randomUnitSphereVec3(seed)) * 1.00));
		ray.direction = payload.world_normal + (normalize(randomUnitSphereVec3(seed)) * 1.00);

		//color = { payload.world_normal.x, payload.world_normal.y, payload.world_normal.z };//debug normals
	}

	color = { sqrtf(color.x),sqrtf(color.y) ,sqrtf(color.z) };//uses 1/gamma=2 not 2.2
	//color = fminf(color, { 1,1,1 });
	return color;
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