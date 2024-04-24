#pragma once
#include "core/Application/private/Layer.hpp"
#include "core/Renderer/Renderer.hpp"

#include <string>

struct Scene;

class EditorLayer : public Layer
{
public:
	__host__ virtual void OnAttach() override;
	__host__ virtual void OnUIRender() override;
	__host__ virtual void OnUpdate(float ts) override;
	__host__ virtual void OnDetach() override;
private:
	struct DevMetrics
	{
		uint32_t m_TrianglesCount = 0;
		uint32_t m_ObjectsCount = 0;
		uint32_t m_MaterialsCount = 0;
	};

	float m_LastFrameTime = 0;//ms
	float m_LastApplicationFrameTime = 0;
	float m_LastRenderTime = 0;//ms
	float2 mousedelta = { 0,0 };
	float3 movedir = { 0,0,0 };
	std::string msg = "Hello World";
private:
	DevMetrics m_DevMetrics;
	Renderer m_Renderer;
	Scene* m_Scene = nullptr;
	Camera* m_dcamera = nullptr;
};