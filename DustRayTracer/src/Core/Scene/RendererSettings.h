#pragma once
#include <vector_types.h>

struct RendererSettings
{
	enum class RenderModes
	{
		NORMALMODE = 0,
		DEBUGMODE = 1
	};

	enum class DebugModes
	{
		ALBEDO_DEBUG = 0,
		NORMAL_DEBUG = 1,
		BARYCENTRIC_DEBUG = 2,
		UVS_DEBUG = 3,
		MESHBVH_DEBUG = 4,
		WORLDBVH_DEBUG = 5
	};

	float global_reflectance = 0.5f;
	float global_metallic = 0;
	float global_roughness = 1.f;
	float3 global_albedo = { 1,0,0 };
	bool use_global_basecolor = false;

	bool gamma_correction = true;
	bool tone_mapping = true;
	bool enableSunlight = false;
	int max_samples = 500;
	int ray_bounce_limit = 2;
	RenderModes RenderMode = RenderModes::NORMALMODE;
	DebugModes DebugMode = DebugModes::ALBEDO_DEBUG;
	float2 sunlight_dir = { -0.803f,0.681f };
	float3 sunlight_color = { 1.000f,0.944f,0.917f };
	float sunlight_intensity = 30;
	float3 sky_color = { 0.25f ,0.498f,0.80f };
	//float3 sky_color = { 0.609,0.695,1.000 }; //apparently physically based?
	float sky_intensity = 10;
};
