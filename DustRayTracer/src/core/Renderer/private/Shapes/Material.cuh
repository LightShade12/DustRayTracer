#pragma once
#include <vector_types.h>

struct Material
{
	Material() = default;
	Material(float3 albedo) :Albedo(albedo) {};
	float3 Albedo = { 1,1,1 };
	int AlbedoTextureIndex = -1;
	float Roughness = 0.8f;
};