#pragma once
#include "private/Layer.hpp"

#include <string>
#include <vector>
#include <memory>



struct GLFWwindow;

struct ApplicationSpecification
{
	std::string Name = "Window01";
	uint32_t Width = 1600;
	uint32_t Height = 900;
};

class Application
{
public:

	Application(const ApplicationSpecification& applicationSpecification = ApplicationSpecification());
	static Application& Get();
	void Close();
	float GetTime();
	float GetFrameTime() const { return m_FrameTime; }
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
	const char* m_GLSL_Version = "#version 460";
private:
	void Init();
	void Shutdown();
private:
	ApplicationSpecification m_Specification;
	GLFWwindow* m_WindowHandle = nullptr;
	bool m_Running = false;

	float m_TimeStep = 0.0f;
	float m_FrameTime = 0.0f;
	float m_LastFrameTime = 0.0f;

	std::vector<std::shared_ptr<Layer>> m_LayerStack;
};