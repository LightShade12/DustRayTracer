#pragma once
#include "Application/private/Layer.hpp"

#include "Core/Renderer.hpp"

#include "Panels/RendererMetricsPanel.hpp"
#include "Panels/MaterialManagerPanel.hpp"

//#include "DustRayTracer/include/DustRayTracer.hpp"

#include <memory>
#include <vector>
#include <string>

struct Scene;

class EditorLayer : public Layer
{
public:
	virtual void OnAttach() override;
	virtual void OnUIRender() override;
	virtual void OnUpdate(float ts) override;
	virtual void OnDetach() override;
	std::vector<const char*> ConsoleLogs;
private:
	bool saveImage(const char* filename, int _width, int _height, GLubyte* data);

private:
	MaterialManagerPanel m_MaterialManagerPanel;
	RendererMetricsPanel m_RendererMetricsPanel;
	float m_LastFrameTime_ms = 0;
	float m_LastRenderTime_ms = 0;
	std::string test_window_text = "Hello World";
	Renderer m_Renderer;
	std::shared_ptr<Scene> m_Scene = nullptr;
	Camera* m_device_Camera = nullptr;
};