#pragma once
#include <float.h>
#include <stdint.h>

namespace DustRayTracer {
	class PathTracerRenderer;
	class HostCamera;
}

class RendererMetricsPanel
{
public:
	RendererMetricsPanel() = default;
	RendererMetricsPanel(const DustRayTracer::PathTracerRenderer& renderer, const DustRayTracer::HostCamera* camera) : m_Renderer(&renderer), DeviceCamera(camera) {};
	void SetRenderer(const DustRayTracer::PathTracerRenderer& renderer) { m_Renderer = &renderer; }
	void SetCamera(const DustRayTracer::HostCamera* camera) { DeviceCamera = camera; }
	void OnUIRender(float last_frame_time_ms, float last_render_time_ms);
	~RendererMetricsPanel();
	struct DevMetrics
	{
		uint32_t m_TrianglesCount = 0;
		uint32_t m_ObjectsCount = 0;
		uint32_t m_MaterialsCount = 0;
		uint32_t m_TexturesCount = 0;
	};
	DevMetrics m_DevMetrics;
private:
	bool skip = true;
	int framecounter = 0;
	float renderfreqmin = FLT_MAX, renderfreqmax = 0, renderfreqavg = 0, renderfreq = 0, rendercumulation = 0;

	const DustRayTracer::PathTracerRenderer* m_Renderer=nullptr;

	const DustRayTracer::HostCamera* DeviceCamera=nullptr;
};