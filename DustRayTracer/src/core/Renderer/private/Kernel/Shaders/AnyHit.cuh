#pragma once
#include "core/Renderer/private/Shapes/triangle.cuh"
#include "core/Renderer/private/Kernel/HitPayload.cuh"
#include "core/Renderer/private/Kernel/Ray.cuh"
#include "core/Renderer/private/Shapes/Mesh.cuh"
#include "core/Renderer/private/CudaMath/helper_math.cuh"

__device__ bool AnyHit(const Ray& ray, const SceneData* scenedata, const Mesh* mesh, const Triangle* triangle, float hit_distance)
{
	float alphaval = 1;
	HitPayload payload;

	const Material material = scenedata->DeviceMaterialBufferPtr[mesh->m_dev_triangles[0].MaterialIdx];
	if (material.AlbedoTextureIndex < 0)
		return true;
	const Texture tex = scenedata->DeviceTextureBufferPtr[material.AlbedoTextureIndex];

	if (tex.componentCount < 4)
		return true;

	float3 hitpoint = ray.origin + ray.direction * hit_distance;

	float3 v0v1 = triangle->vertex1.position - triangle->vertex0.position;
	float3 v0v2 = triangle->vertex2.position - triangle->vertex0.position;
	float3 v0p = hitpoint - triangle->vertex0.position;

	float d00 = dot(v0v1, v0v1);
	float d01 = dot(v0v1, v0v2);
	float d11 = dot(v0v2, v0v2);
	float d20 = dot(v0p, v0v1);
	float d21 = dot(v0p, v0v2);

	float denom = d00 * d11 - d01 * d01;
	payload.UVW.y = (d11 * d20 - d01 * d21) / denom;
	payload.UVW.z = (d00 * d21 - d01 * d20) / denom;
	payload.UVW.x = 1.0f - payload.UVW.y - payload.UVW.z;

	float2 uv = {
				 payload.UVW.x * triangle->vertex0.UV.x + payload.UVW.y * triangle->vertex1.UV.x + payload.UVW.z * triangle->vertex2.UV.x,
				  payload.UVW.x * triangle->vertex0.UV.y + payload.UVW.y * triangle->vertex1.UV.y + payload.UVW.z * triangle->vertex2.UV.y
	};
	alphaval = tex.getAlpha(uv);

	if (alphaval < 1)return false;

	return true;
}