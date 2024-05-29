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