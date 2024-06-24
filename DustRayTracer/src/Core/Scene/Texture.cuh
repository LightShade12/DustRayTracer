#pragma once
#include <vector_types.h>

class Texture
{
public:
	Texture() = default;
	Texture(const char* filepath);
	Texture(const unsigned char* data, size_t bytesize);
	__device__ float3 getPixel(float2 UV, bool noncolor = false) const;
	__device__ float getAlpha(float2 UV) const;
	void Cleanup();

	char Name[32] = "unnamed";
	int width, height = 0;
	int componentCount = 0;
	int ChannelBitDepth = 0;
	//bool isfloatingPoint=false;
	unsigned char* d_data = nullptr;//device resident
private:
};