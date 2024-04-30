#include "core/Renderer/private/Kernel/HitPayload.cuh"

__device__ HitPayload Debug()
{
	HitPayload payload;
	payload.debug = true;
	return payload;
}