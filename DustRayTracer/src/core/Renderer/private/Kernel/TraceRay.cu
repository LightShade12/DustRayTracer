#include "TraceRay.cuh"

#include "core/Renderer/private/Shapes/Scene.cuh"
#include "core/Renderer/private/Kernel/BVH.cuh"

#include "Shaders/ClosestHit.cuh"
#include "Shaders/Miss.cuh"
#include "Shaders/Intersection.cuh"
#include "Shaders/AnyHit.cuh"

#include <vector_types.h>

//traverse accel struct
__device__ HitPayload TraceRay(const Ray& ray, const SceneData* scenedata) {
	int closestObjectIdx = -1;
	int hitTriangleIdx = -1;
	float hitDistance = FLT_MAX;
	HitPayload workingPayload;

	//top level
	if (scenedata->DeviceBVHTreePtr->IntersectAABB(ray))
	{
		for (int nodeIdx = 0; nodeIdx < scenedata->DeviceBVHTreePtr->childrenCount; nodeIdx++)
		{
			const Node currentnode = scenedata->DeviceBVHTreePtr->children[nodeIdx];

			//2nd level
			if (currentnode.IntersectAABB(ray))
			{
				const Mesh* currentmesh = currentnode.d_Mesh;

				for (int triangleIdx = 0; triangleIdx < currentmesh->m_trisCount; triangleIdx++)
				{
					const Triangle* triangle = &(currentmesh->m_dev_triangles[triangleIdx]);
					workingPayload = Intersection(ray, triangle);

					if (workingPayload.hit_distance < hitDistance && workingPayload.hit_distance>0)
					{
						if (!AnyHit(ray, scenedata,
							currentmesh, triangle, workingPayload.hit_distance))continue;
						hitDistance = workingPayload.hit_distance;
						closestObjectIdx = currentnode.MeshIndex;
						hitTriangleIdx = triangleIdx;
					}
				}
			}
		}
	}

	if (false)
		for (size_t meshIdx = 0; meshIdx < scenedata->DeviceMeshBufferSize; meshIdx++)
		{
			const Mesh* currentmesh = &(scenedata->DeviceMeshBufferPtr[meshIdx]);

			for (int triangleIdx = 0; triangleIdx < currentmesh->m_trisCount; triangleIdx++)
			{
				const Triangle* triangle = &(currentmesh->m_dev_triangles[triangleIdx]);
				workingPayload = Intersection(ray, triangle);

				if (workingPayload.hit_distance < hitDistance && workingPayload.hit_distance>0)
				{
					if (!AnyHit(ray, scenedata,
						currentmesh, triangle, workingPayload.hit_distance))continue;
					hitDistance = workingPayload.hit_distance;
					closestObjectIdx = meshIdx;
					hitTriangleIdx = triangleIdx;
				}
			}
		}

	//Have not hit
	if (closestObjectIdx < 0)
	{
		return Miss(ray);
	}

	return ClosestHit(ray, closestObjectIdx, hitDistance, scenedata->DeviceMeshBufferPtr, hitTriangleIdx);
}

//does not support glass material; cuz no mat processing
__device__ bool RayTest(const Ray& ray, const SceneData* scenedata)
{
	HitPayload workingPayload;

	for (size_t meshIdx = 0; meshIdx < scenedata->DeviceMeshBufferSize; meshIdx++)
	{
		const Mesh* currentmesh = &(scenedata->DeviceMeshBufferPtr[meshIdx]);
		//loop over mesh and triangles
		for (int triangleIdx = 0; triangleIdx < currentmesh->m_trisCount; triangleIdx++)
		{
			const Triangle* triangle = &(currentmesh->m_dev_triangles[triangleIdx]);
			workingPayload = Intersection(ray, triangle);

			//if intersecting
			if (workingPayload.hit_distance > 0)
			{
				//if opaque
				if (AnyHit(ray, scenedata,
					currentmesh, triangle, workingPayload.hit_distance))
					return true;
			}
		}
	}

	return false;
}