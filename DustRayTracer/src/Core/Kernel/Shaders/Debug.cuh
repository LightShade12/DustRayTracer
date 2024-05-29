#include "Core/HitPayload.cuh"

__device__ HitPayload Debug()
{
	HitPayload payload;
	payload.debug = true;
	return payload;
}