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
		MESHBVH_DEBUG =4,
		WORLDBVH_DEBUG =5
	};

	bool gamma_correction = false;
	bool enableSunlight = false;
	int max_samples = 50;
	int ray_bounce_limit = 5;
	RenderModes RenderMode = RenderModes::NORMALMODE;
	DebugModes DebugMode = DebugModes::ALBEDO_DEBUG;
	float2 sunlight_dir = { 0.83,0.41 };
	float3 sunlight_color = { 1.000,0.944,0.917 };
	float sunlight_intensity = 1;
	float3 sky_color = { 0.122,0.341,1.0 };
	float sky_intensity = 1.5;
};
