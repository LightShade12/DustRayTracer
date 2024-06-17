#pragma once
#include "Application/private/Layer.hpp"

#include "Core/Renderer.hpp"
//#include "DustRayTracer/include/DustRayTracer.hpp"

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
	struct DevMetrics
	{
		uint32_t m_TrianglesCount = 0;
		uint32_t m_ObjectsCount = 0;
		uint32_t m_MaterialsCount = 0;
		uint32_t m_TexturesCount = 0;
	};

	float m_LastFrameTime_ms = 0;//ms
	//float m_LastApplicationFrameTime = 0;
	float m_LastRenderTime_ms = 0;//ms
	std::string test_window_text = "Hello World";

private:
	bool skip = true;
	float renderfreqmin = FLT_MAX, renderfreqmax = 0, renderfreqavg = 0, renderfreq = 0, rendercumulation = 0;
	int framecounter = 0;
	DevMetrics m_DevMetrics;
	Renderer m_Renderer;
	Scene* m_Scene = nullptr;
	Camera* m_device_Camera = nullptr;
};