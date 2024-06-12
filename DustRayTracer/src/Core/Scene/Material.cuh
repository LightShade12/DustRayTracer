#pragma once
#include <vector_types.h>

struct Material
{
	//std::string name;
	Material() = default;
	Material(float3 albedo) :Albedo(albedo) {};

	float3 Albedo = { 1,1,1 };
	float3 EmmisiveFactor = { 0,0,0 };

	int AlbedoTextureIndex = -1;
	//int NormalTextureIndex = -1;
	//float NormalScale=1;
	//int RoughnessTextureIndex = -1;
	//int EmissionTextureIndex = -1;

	float Roughness = 0.f;
	bool Transmission = false;
	float refractive_index = 1.45f;
	bool Metallic = false;
};