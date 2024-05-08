#include "Texture.cuh"

#include "stb_image.h"

#include "core/Editor/Common/CudaCommon.cuh"

//8 bit img only
Texture::Texture(const char* filepath)
{
	unsigned char* imgdata = stbi_load(filepath, &width, &height, &componentCount, 0);
	size_t imgbuffersize = width * height * componentCount * sizeof(unsigned char);//1byte ik; till float texture support, this is format
	cudaMallocManaged((void**)&d_data, imgbuffersize);
	checkCudaErrors(cudaGetLastError());
	cudaMemcpy(d_data, imgdata, imgbuffersize, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
	stbi_image_free(imgdata);
}

Texture::Texture(unsigned char* data, size_t bytesize)
{
	unsigned char* imgdata = stbi_load_from_memory(data, bytesize, &width, &height, &componentCount, 0);
	size_t imgbuffersize = width * height * componentCount * sizeof(unsigned char);//1byte ik; till float texture support, this is format
	cudaMallocManaged((void**)&d_data, imgbuffersize);
	checkCudaErrors(cudaGetLastError());
	cudaMemcpy(d_data, imgdata, imgbuffersize, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
	stbi_image_free(imgdata);
}

__device__ float3 Texture::getPixel(float2 UV) const
{
	int x = UV.x * width;
	int y = UV.y * height;

	uchar4 fcol = { 255,0,255,255 };
	if (componentCount == 3)
	{
		uchar3* coldata = (uchar3*)d_data;
		uchar3 rgb = (coldata[y * width + x]);//bad var name
		fcol = { rgb.x,rgb.y,rgb.z };
	}
	else if (componentCount == 4)
	{
		uchar4* coldata = (uchar4*)d_data;
		fcol = (coldata[y * width + x]);
	}

	//printf("r: %u g: %u b: %u ||", (unsigned int)fcol.x, (unsigned int)fcol.y, (unsigned int)fcol.z);

	float3 fltcol = make_float3(fcol.x / (float)255, fcol.y / (float)255, fcol.z / (float)255);
	//printf("r: %.3f g: %.3f b: %.3f ||", fltcol.x, fltcol.y, fltcol.z);

	return fltcol;
}

__device__ float Texture::getAlpha(float2 UV) const
{
	if (componentCount < 4)
		return 1;

	uchar4* coldata = (uchar4*)d_data;
	int x = UV.x * width;
	int y = UV.y * height;
	unsigned char fcol = (coldata[y * width + x]).w;
	//printf("r: %u g: %u b: %u ||", (unsigned int)fcol.x, (unsigned int)fcol.y, (unsigned int)fcol.z);

	float alpha = fcol / (float)255;

	return alpha;
}

void Texture::Cleanup()
{
	cudaFree(d_data);
	checkCudaErrors(cudaGetLastError());
}