#pragma once
#include <vector_types.h>

class Texture
{
public:
	Texture() = default;
	Texture(const char* filepath);
	Texture(unsigned char* data, size_t bytesize);
	__device__ float3 getPixel(float2 UV) const;
	__device__ float getAlpha(float2 UV) const;
	void Cleanup();

	std::string name;
	int width, height = 0;
	int componentCount = 0;
	//bool isfloatingPoint=false;
private:
	unsigned char* d_data = nullptr;
};