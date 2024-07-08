#include "Core/HitPayload.cuh"

__device__ HitPayload Debug()
{
	HitPayload out_payload;
	out_payload.debug = true;
	return out_payload;
}