#include "TraceRay.cuh"

#include "core/Renderer/private/Shapes/Scene.cuh"
#include "core/Renderer/private/Kernel/BVH.cuh"

#include "Shaders/ClosestHit.cuh"
#include "Shaders/Miss.cuh"
#include "Shaders/Intersection.cuh"
#include "Shaders/AnyHit.cuh"
#include "Shaders/Debug.cuh"

#include <vector_types.h>

//traverse accel struct
__device__ HitPayload TraceRay(const Ray& ray, const SceneData* scenedata) {
	float hitDistance = FLT_MAX;//closesthit
	HitPayload workingPayload;
	const Triangle* hitprim = nullptr;

	for (int triangleIdx = 0; triangleIdx < scenedata->DevicePrimitivesBufferSize; triangleIdx++)
	{
		const Triangle* triangle = &(scenedata->DevicePrimitivesBuffer[triangleIdx]);
		workingPayload = Intersection(ray, triangle);

		if (workingPayload.hit_distance < hitDistance && workingPayload.hit_distance>0)
		{
			if (!AnyHit(ray, scenedata,
				triangle, workingPayload.hit_distance))continue;
			hitDistance = workingPayload.hit_distance;
			hitprim = workingPayload.primitiveptr;
		}
	}

	//Have not hit
	if (hitprim == nullptr)
	{
		return Miss(ray);
	}

	return ClosestHit(ray, hitDistance, hitprim);
}

//does not support glass material; cuz no mat processing
__device__ bool RayTest(const Ray& ray, const SceneData* scenedata)
{
	HitPayload workingPayload;

	//loop over mesh and triangles
	for (int triangleIdx = 0; triangleIdx < scenedata->DevicePrimitivesBufferSize; triangleIdx++)
	{
		const Triangle* triangle = &(scenedata->DevicePrimitivesBuffer[triangleIdx]);
		workingPayload = Intersection(ray, triangle);

		//if intersecting
		if (workingPayload.hit_distance > 0)
		{
			//if opaque
			if (AnyHit(ray, scenedata,
				triangle, workingPayload.hit_distance))
				return true;
		}
	}

	return false;
}