#pragma once
#include "Core/Scene/triangle.cuh"
#include "Core/HitPayload.cuh"
#include "Core/Ray.cuh"
#include "Core/Scene/Mesh.cuh"
#include "Core/CudaMath/helper_math.cuh"

__device__ bool AnyHit(const Ray& ray, const SceneData* scenedata, const ShortHitPayload* in_payload)
{
	const Material* material = &(scenedata->DeviceMaterialBufferPtr[in_payload->primitiveptr->material_idx]);

	if (material->AlbedoTextureIndex < 0)
		return true;

	const Texture* tex = &(scenedata->DeviceTextureBufferPtr[material->AlbedoTextureIndex]);

	if (tex->componentCount < 4)
		return true;

	float2 UV = in_payload->UVW.x * in_payload->primitiveptr->vertex0.UV +
		in_payload->UVW.y * in_payload->primitiveptr->vertex1.UV +
		in_payload->UVW.z * in_payload->primitiveptr->vertex2.UV;

	float alphaval = 1;
	alphaval = tex->getAlpha(UV);

	return (!(alphaval < 1));
}