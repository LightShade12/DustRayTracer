#pragma once
#include <vector_types.h>
#include <string>

struct Material
{
public:
	Material() = default;
	Material(float3 albedo) :Albedo(albedo) {};

public:
	char Name[32];
	float3 Albedo = { 1,1,1 };
	float3 EmissiveColor = { 0,0,0 };
	int AlbedoTextureIndex = -1;
	int RoughnessTextureIndex = -1;
	int NormalTextureIndex = -1;
	int EmissionTextureIndex = -1;
	float Reflectance = 0.5;// default: F0=0.16
	float Metallicity = 0;
	float Roughness = 0.8f;
	float NormalMapScale = 1.0f;
};