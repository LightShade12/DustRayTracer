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

	bool gamma_correction = true;
	bool enableSunlight = false;
	int max_samples = 6;
	int ray_bounce_limit = 2;
	RenderModes RenderMode = RenderModes::DEBUGMODE;
	DebugModes DebugMode = DebugModes::ALBEDO_DEBUG;
	float2 sunlight_dir = { 0.83f,0.41f };
	float3 sunlight_color = { 1.000f,0.944f,0.917f };
	float sunlight_intensity = 1;
	float3 sky_color = { 0.122f,0.341f,1.0f };
	float sky_intensity = 1.5;
};
