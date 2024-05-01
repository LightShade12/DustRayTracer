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
		UVS_DEBUG = 3
	};

	bool gamma_correction = false;
	bool enableSunlight = false;
	int max_samples = 50;
	int ray_bounce_limit = 5;
	RenderModes RenderMode = RenderModes::NORMALMODE;
	DebugModes DebugMode = DebugModes::NORMAL_DEBUG;
	float3 sunlight_dir = { 1,1,1 };
	float3 sunlight_color = { 1.000,0.944,0.917 };
	float sunlight_intensity = 1;
	float3 sky_color = { 0.2,0.5,1.0 };
	float sky_intensity = 1;
};
