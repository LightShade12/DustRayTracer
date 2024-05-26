#include "TraceRay.cuh"

#include "core/Renderer/private/Shapes/Scene.cuh"
#include "core/Renderer/private/Kernel/BVH/BVHTraversal.cuh"

#include "Shaders/ClosestHit.cuh"
#include "Shaders/Miss.cuh"
#include "Shaders/Intersection.cuh"
#include "Shaders/AnyHit.cuh"
#include "Shaders/Debug.cuh"

#include <vector_types.h>

//traverse accel struct
__device__ HitPayload TraceRay(const Ray& ray, const SceneData* scenedata) {
	bool useBVH = true;
	float closestHitDistance = FLT_MAX;//closesthit
	HitPayload workingPayload;
	const Triangle* hitprim = nullptr;
	bool debug = false;

	if (useBVH) {
		//here working payload is being sent in as closest hit payload
		workingPayload.hit_distance = FLT_MAX;
		find_closest_hit_iterative(ray, scenedata->DeviceBVHTreeRootPtr, &workingPayload, debug, scenedata);
		closestHitDistance = workingPayload.hit_distance;
		hitprim = workingPayload.primitiveptr;
	}
	else {
		for (int triangleIdx = 0; triangleIdx < scenedata->DevicePrimitivesBufferSize; triangleIdx++)
		{
			const Triangle* triangle = &(scenedata->DevicePrimitivesBuffer[triangleIdx]);
			workingPayload = Intersection(ray, triangle);

			if (workingPayload.hit_distance < closestHitDistance && workingPayload.hit_distance>0)
			{
				if (!AnyHit(ray, scenedata,
					triangle, workingPayload.hit_distance))continue;
				closestHitDistance = workingPayload.hit_distance;
				hitprim = workingPayload.primitiveptr;
			}
		}
	}

	if (debug)
		return Debug();

	//Have not hit
	if (hitprim == nullptr)
	{
		return Miss(ray, workingPayload.color);
	}

	return ClosestHit(ray, closestHitDistance, hitprim, workingPayload.color);
}

//does not support glass material; cuz no mat processing
__device__ bool RayTest(const Ray& ray, const SceneData* scenedata)
{
	bool useBVH = true;
	HitPayload workingPayload;
	bool debug = false;
	if (useBVH) {
		return RayTest_bvh(ray, scenedata->DeviceBVHTreeRootPtr, scenedata);
	}
	else
	{
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
}