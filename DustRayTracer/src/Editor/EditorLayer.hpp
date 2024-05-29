#pragma once
#include "Application/private/Layer.hpp"
#include "Core/Renderer.hpp"
#include <vector>

#include <string>

struct Scene;

class EditorLayer : public Layer
{
public:
	__host__ virtual void OnAttach() override;
	__host__ virtual void OnUIRender() override;
	__host__ virtual void OnUpdate(float ts) override;
	__host__ virtual void OnDetach() override;
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
	DevMetrics m_DevMetrics;
	Renderer m_Renderer;
	Scene* m_Scene = nullptr;
	Camera* m_device_Camera = nullptr;
};