#pragma once
#include "private/Layer.hpp"

#include <string>
#include <vector>
#include <memory>

struct GLFWwindow;

struct ApplicationSpecification
{
	std::string Name = "DustRayTracer";
	uint32_t Width = 1600;
	uint32_t Height = 900;
};

class Application
{
public:
	Application(const ApplicationSpecification& applicationSpecification = ApplicationSpecification());
	static Application& Get();
	void Close();
	float GetTimeSeconds();//TODO: ambigious terms
	float GetFrameTimeSecs() const { return m_FrameTimeSecs; }
	void Run();
	template<typename T>
	void PushLayer()
	{
		static_assert(std::is_base_of<Layer, T>::value, "Pushed type is not subclass of Layer!");
		m_LayerStack.emplace_back(std::make_shared<T>())->OnAttach();
	}

	void PushLayer(const std::shared_ptr<Layer>& layer) { m_LayerStack.emplace_back(layer); layer->OnAttach(); }

	GLFWwindow* GetWindowHandle() const { return m_WindowHandle; }

	~Application();
private:
	void Init();
	void Shutdown();
private:
	const char* m_GLSL_VERSION = "#version 460";
	ApplicationSpecification m_Specification;
	GLFWwindow* m_WindowHandle = nullptr;
	bool m_Running = false;

	float m_TimeStepSecs = 0.0f;//TODO: is this correct unit?
	float m_FrameTimeSecs = 0.0f;
	float m_LastFrameTimeSecs = 0.0f;

	std::vector<std::shared_ptr<Layer>> m_LayerStack;
};