#include "Core/CudaMath/Random.cuh"
#include <cstdint>
#include <vector_types.h>

class Sampler {
	virtual void StartSampler(float2 pixel, uint32_t sampleidx, int dimension) = 0;//dim will replace bounces
	virtual float Get1DSample() = 0;
	virtual float2 Get2DSample() = 0;
	virtual float2 GetPixel2D() = 0;
};

class PCGSampler : public Sampler {
};