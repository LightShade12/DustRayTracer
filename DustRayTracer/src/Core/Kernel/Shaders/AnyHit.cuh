#pragma once
#include "Core/Scene/triangle.cuh"
#include "Core/HitPayload.cuh"
#include "Core/Ray.cuh"
#include "Core/Scene/Scene.cuh"
#include "Core/CudaMath/helper_math.cuh"

__device__ bool AnyHit(const Ray& ray, const SceneData* scene_data, const ShortHitPayload* in_payload)
{
	const Triangle& triangle = scene_data->DevicePrimitivesBuffer[in_payload->triangle_idx];

	const DustRayTracer::MaterialData* material = &(scene_data->DeviceMaterialBufferPtr[triangle.material_idx]);

	if (material->AlbedoTextureIndex < 0)
		return true;

	const Texture* tex = &(scene_data->DeviceTextureBufferPtr[material->AlbedoTextureIndex]);

	if (tex->componentCount < 4)
		return true;

	float2 UV = in_payload->UVW.x * triangle.vertex0.UV +
		in_payload->UVW.y * triangle.vertex1.UV +
		in_payload->UVW.z * triangle.vertex2.UV;

	float alphaval = 1;
	alphaval = tex->getAlpha(UV);

	return (!(alphaval < 1));
}