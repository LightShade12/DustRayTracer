#include "Bounds.cuh"
#include "Core/CudaMath/helper_math.cuh"

float Bounds3f::getSurfaceArea() const
{
	float planex = 2 * (pMax.z - pMin.z) * (pMax.y - pMin.y);
	float planey = 2 * (pMax.z - pMin.z) * (pMax.x - pMin.x);
	float planez = 2 * (pMax.x - pMin.x) * (pMax.y - pMin.y);
	return planex + planey + planez;
}

float3 Bounds3f::getCentroid() const
{
	return 0.5f * pMin + 0.5f * pMax;
}

__device__ float Bounds3f::intersect(const Ray& ray) const
{
	float3 t0 = (pMin - ray.getOrigin()) * ray.getInvDir();
	float3 t1 = (pMax - ray.getOrigin()) * ray.getInvDir();

	float3 tmin = fminf(t0, t1);
	float3 tmax = fmaxf(t1, t0);//switched order of t to guard NaNs

	//min max componenet
	float tenter = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
	float texit = fminf(fminf(tmax.x, tmax.y), tmax.z);

	// Adjust tenter if the ray starts inside the AABB
	if (tenter < 0.0f) {
		tenter = 0.0f;
	}

	if (tenter > texit || texit < 0) {
		return -1; // No intersection
	}

	return tenter;
}