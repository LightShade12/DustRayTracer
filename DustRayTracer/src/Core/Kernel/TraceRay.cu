#include "TraceRay.cuh"

#include "Core/Scene/Scene.cuh"
#include "Core/BVH/BVHTraversal.cuh"

#include "Shaders/ClosestHit.cuh"
#include "Shaders/Miss.cuh"
#include "Shaders/Intersection.cuh"
#include "Shaders/AnyHit.cuh"
#include "Shaders/Debug.cuh"

#include <vector_types.h>

//traverse accel struct
__device__ HitPayload traceRay(const Ray& ray, const SceneData* scenedata) {
	HitPayload workingPayload;

	workingPayload.hit_distance = ray.interval.max;//this prevents closesthit being always closest
	//here working payload is being sent in as closest hit payload
	traverseBVH(ray, (scenedata->DeviceBVHNodesBufferSize) - 1, &workingPayload, scenedata);

	//if (debug)
		//return Debug();

	//Have not hit
	if (workingPayload.triangle_idx == -1)
	{
		return Miss(ray, workingPayload.color);
	}

	return ClosestHit(ray, &workingPayload, scenedata);
}

//does not support glass material; cuz no mat processing
__device__ int rayTest(const Ray& ray, const SceneData* scenedata)
{
	return traverseBVH_raytest(ray, (scenedata->DeviceBVHNodesBufferSize) - 1, scenedata);
}