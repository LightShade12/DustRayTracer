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
	float closestHitDistance = FLT_MAX;//closesthit
	HitPayload workingPayload;
	const Triangle* hitprim = nullptr;
	bool debug = false;

	//here working payload is being sent in as closest hit payload
	workingPayload.hit_distance = FLT_MAX;
	traverseBVH(ray, scenedata->DeviceBVHTreeRootPtr, &workingPayload, debug, scenedata);
	closestHitDistance = workingPayload.hit_distance;
	hitprim = workingPayload.primitiveptr;

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
	return traverseBVH_raytest(ray, scenedata->DeviceBVHTreeRootPtr, scenedata);
}