#include "Bounds.cuh"

float Bounds3f::getArea()
{
	float planex = 2 * (pMax.z - pMin.z) * (pMax.y - pMin.y);
	float planey = 2 * (pMax.z - pMin.z) * (pMax.x - pMin.x);
	float planez = 2 * (pMax.x - pMin.x) * (pMax.y - pMin.y);
	return planex + planey + planez;
}