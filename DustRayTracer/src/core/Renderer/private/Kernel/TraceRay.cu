#include "TraceRay.cuh"

#include "core/Renderer/private/Shapes/Scene.cuh"

#include "Shaders/ClosestHit.cuh"
#include "Shaders/Miss.cuh"
#include "Shaders/Intersection.cuh"

#include <vector_types.h>

//traverse accel struct
__device__ HitPayload TraceRay(const Ray& ray, const Mesh* MeshBufferPtr, size_t MeshBufferSize) {
	int closestObjectIdx = -1;
	int hitTriangleIdx = -1;
	float hitDistance = FLT_MAX;
	HitPayload workingPayload;

	for (size_t meshIdx = 0; meshIdx < MeshBufferSize; meshIdx++)
	{
		const Mesh* currentmesh = &(MeshBufferPtr[meshIdx]);

		for (int triangleIdx = 0; triangleIdx < currentmesh->m_trisCount; triangleIdx++)
		{
			const Triangle* triangle = &(currentmesh->m_dev_triangles[triangleIdx]);
			workingPayload = Intersection(ray, triangle);

			if (workingPayload.hit_distance < hitDistance && workingPayload.hit_distance>0)
			{
				hitDistance = workingPayload.hit_distance;
				closestObjectIdx = meshIdx;
				hitTriangleIdx = triangleIdx;
			}
		}
	}

	if (closestObjectIdx < 0)
	{
		return Miss(ray);
	}

	return ClosestHit(ray, closestObjectIdx, hitDistance,MeshBufferPtr, hitTriangleIdx);
}