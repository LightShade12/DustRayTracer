#pragma once
#include "Core/Scene/triangle.cuh"
#include "Core/HitPayload.cuh"
#include "Core/Ray.cuh"
#include "Core/Scene/Mesh.cuh"
#include "Core/CudaMath/helper_math.cuh"

__device__ bool AnyHit(const Ray& ray, const SceneData* scenedata, const Triangle* triangle, float hit_distance)
{
	const Material* material = &(scenedata->DeviceMaterialBufferPtr[triangle->materialIdx]);
	if (material->AlbedoTextureIndex < 0)
		return true;
	const Texture* tex = &(scenedata->DeviceTextureBufferPtr[material->AlbedoTextureIndex]);

	if (tex->componentCount < 4)
		return true;

	float3 hitpoint = ray.getOrigin() + ray.getDirection() * hit_distance;

	//HitPayload payload;

	float3 v0v1 = triangle->vertex1.position - triangle->vertex0.position;
	float3 v0v2 = triangle->vertex2.position - triangle->vertex0.position;
	float3 v0p = hitpoint - triangle->vertex0.position;

	float d00 = dot(v0v1, v0v1);
	float d01 = dot(v0v1, v0v2);
	float d11 = dot(v0v2, v0v2);
	float d20 = dot(v0p, v0v1);
	float d21 = dot(v0p, v0v2);

	float denom = d00 * d11 - d01 * d01;
	float3 UVW;
	UVW.y = (d11 * d20 - d01 * d21) / denom;
	UVW.z = (d00 * d21 - d01 * d20) / denom;
	UVW.x = 1.0f - UVW.y - UVW.z;

	float2 uv = {
				 UVW.x * triangle->vertex0.UV.x + UVW.y * triangle->vertex1.UV.x + UVW.z * triangle->vertex2.UV.x,
				 UVW.x * triangle->vertex0.UV.y + UVW.y * triangle->vertex1.UV.y + UVW.z * triangle->vertex2.UV.y
	};
	float alphaval = 1;
	alphaval = tex->getAlpha(uv);

	if (alphaval < 1)return false;

	return true;
}