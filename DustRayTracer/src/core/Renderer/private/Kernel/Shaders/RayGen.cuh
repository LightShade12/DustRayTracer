#pragma once
#include "core/Renderer/private/Kernel/TraceRay.cuh"

#include <core/Renderer/private/CudaMath/helper_math.cuh>
#include <core/Renderer/private/CudaMath/Random.cuh>

#include <core/Renderer/private/Kernel/Ray.cuh>
#include <core/Renderer/private/Kernel/HitPayload.cuh>
#include <core/Renderer/private/Shapes/Scene.cuh>

__device__ float3 RayGen(uint32_t x, uint32_t y, uint32_t max_x, uint32_t max_y,
	const Camera* cam, const Material* matvector, uint32_t frameidx,
	const Mesh* MeshBufferPtr, size_t MeshBufferSize, const Texture* TextureBufferPtr, size_t TextureBufferSize) {
	float2 uv = { (float(x) / max_x) ,(float(y) / max_y) };

	float3 sunpos = { 100,100,100 };
	float3 suncol = { 1.0, 1.0, 0.584 };
	//uv.x *= ((float)max_x / (float)max_y);
	//uv.x = uv.x * 2.f - ((float)max_x / (float)max_y);
	//uv.y = uv.y * 2.f - 1.f;
	uv = uv * 2 - 1;

	Ray ray;
	ray.origin = cam->m_Position;
	ray.direction = cam->GetRayDir(uv, 30, max_x, max_y);
	float3 light = { 0,0,0 };

	uint32_t seed = x + y * max_x;
	seed *= frameidx;

	float3 contribution = { 1,1,1 };
	int bounces = 10;

	for (int i = 0; i < bounces; i++)
	{
		float2 uv = {0,1};//DEBUG
		HitPayload payload = TraceRay(ray, MeshBufferPtr, MeshBufferSize);
		seed += i;
		//sky
		if (payload.hit_distance < 0)
		{
			float a = 0.5 * (1 + (normalize(ray.direction)).y);
			float3 col1 = { 0.2,0.5,1.0 };
			float3 col2 = { 1,1,1 };
			float3 fcol = (float(1 - a) * col2) + (a * col1);
			light += fcol * contribution;
			break;
		}

		float lightIntensity = max(dot(payload.world_normal, sunpos), 0.0f); // == cos(angle)

		const Mesh closestMesh = MeshBufferPtr[payload.object_idx];
		Material material = matvector[closestMesh.m_dev_triangles[0].MaterialIdx];//TODO: might cause error; maybe not cuz miss shading handles before exec here

		//light = material.Albedo;

		//printf("kernel texture idx eval: %d ", material.AlbedoTextureIndex);

		if (material.AlbedoTextureIndex < 0) 
		{
			contribution *= material.Albedo;
		}
		else
		{
			//printf("exec ");
			Triangle tri = closestMesh.m_dev_triangles[payload.triangle_idx];
			uv = {
				 payload.UVW.x * tri.vertex0.UV.x + payload.UVW.y * tri.vertex1.UV.x + payload.UVW.z * tri.vertex2.UV.x,
				  payload.UVW.x * tri.vertex0.UV.y + payload.UVW.y * tri.vertex1.UV.y + payload.UVW.z * tri.vertex2.UV.y
			};
			contribution *= TextureBufferPtr[material.AlbedoTextureIndex].getPixel(uv);
			//contribution *= {0,1,0};
		}

		float3 newRayOrigin = payload.world_position + (payload.world_normal * 0.0001f);

		if (!RayTest(Ray(newRayOrigin, (sunpos - newRayOrigin) + randomUnitVec3(seed) * 2),
			MeshBufferPtr, MeshBufferSize))
		{
			light += (material.Albedo * suncol) * 0.5;
		}

		ray.origin = newRayOrigin;
		ray.direction = payload.world_normal + (normalize(randomUnitSphereVec3(seed)));

		//light = payload.world_normal;//debug normals
		//light = payload.UVW;//debug barycentric coords
		//light = {uv.x,uv.y,0};//debug UV
	}

	light = { sqrtf(light.x),sqrtf(light.y) ,sqrtf(light.z) };//uses 1/gamma=2 not 2.2
	light = fminf(light, { 1,1,1 });
	return light;
};