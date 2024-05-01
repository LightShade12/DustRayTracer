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
	std::vector<const char*> appLogs;
	Application(const ApplicationSpecification& applicationSpecification = ApplicationSpecification());
	static Application& Get();
	void Close();
	float GetTime_seconds();
	float GetFrameTime_secs() const { return m_FrameTime_secs; }
	void Run();
	void logMessage(const char* msg);
	template<typename T>
	void PushLayer()
	{
		static_assert(std::is_base_of<Layer, T>::value, "Pushed type is not subclass of Layer!");
		m_LayerStack.emplace_back(std::make_shared<T>())->OnAttach();
	}

	void PushLayer(const std::shared_ptr<Layer>& layer) { m_LayerStack.emplace_back(layer); layer->OnAttach(); }

	GLFWwindow* GetWindowHandle() const { return m_WindowHandle; }

	~Application();
	const char* m_GLSL_Version = "#version 460";
private:
	void Init();
	void Shutdown();
private:
	ApplicationSpecification m_Specification;
	GLFWwindow* m_WindowHandle = nullptr;
	bool m_Running = false;

	float m_TimeStep_secs = 0.0f;
	float m_FrameTime_secs = 0.0f;
	float m_LastFrameTime_secs = 0.0f;

	std::vector<std::shared_ptr<Layer>> m_LayerStack;
};