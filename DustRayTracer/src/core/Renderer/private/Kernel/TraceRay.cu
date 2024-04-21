#include "TraceRay.cuh"

#include "core/Renderer/private/Shapes/Scene.cuh"

#include "Shaders/ClosestHit.cuh"
#include "Shaders/Miss.cuh"
#include "Shaders/Intersection.cuh"

#include <vector_types.h>

//traverse accel struct
__device__ HitPayload TraceRay(const Ray& ray, const Triangle* scene_vector, size_t scene_vector_size) {
	int closestObjectIdx = -1;
	float hitDistance = FLT_MAX;
	HitPayload workingPayload;

	for (int i = 0; i < scene_vector_size; i++)
	{
		const Triangle* triangle = &scene_vector[i];
		workingPayload = Intersection(ray, triangle);

		if (workingPayload.hit_distance < hitDistance && workingPayload.hit_distance>0)
		{
			hitDistance = workingPayload.hit_distance;
			closestObjectIdx = i;
		}
	}

	if (closestObjectIdx < 0)
	{
		return Miss(ray);
	}

	return ClosestHit(ray, closestObjectIdx, hitDistance, scene_vector);
};