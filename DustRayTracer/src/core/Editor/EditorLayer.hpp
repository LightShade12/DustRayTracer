#pragma once
#include "core/Application/private/Layer.hpp"
#include "core/Renderer/Renderer.hpp"

#include <string>

struct Scene;

class EditorLayer :	public Layer
{
public:
	EditorLayer();
	__host__ virtual void OnAttach() override;
	__host__ virtual void OnUIRender() override;
	__host__ virtual void OnUpdate(float ts) override;
	__host__ virtual void OnDetach() override;
private:
	float m_LastFrameTime= 0;
	float m_LastApplicationFrameTime = 0;
	float m_LastRenderTime = 0;
	float2 mousedelta = {0,0};
	float3 movedir = {0,0,0};
	std::string msg = "Hello World";
	Renderer m_Renderer;
	Scene* m_Scene=nullptr;
	Camera* m_dcamera = nullptr;
};
