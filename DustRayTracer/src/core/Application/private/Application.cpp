#include "core/Application/Application.hpp"

#include <glad/glad.h>//make sure this is included before cudaGlInterop.h and glfw3.h
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <glm/glm.hpp>

extern bool g_ApplicationRunning;
static Application* s_Instance = nullptr;

static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

Application::Application(const ApplicationSpecification& specification)
	: m_Specification(specification)
{
	s_Instance = this;

	Init();
}

void Application::Run()
{
	m_Running = true;

	//windowloop
	while (!glfwWindowShouldClose(m_WindowHandle) && m_Running)
	{
		//Polling---------------------------------------------------------------------------------
		glfwPollEvents();

		for (auto& layer : m_LayerStack)
			layer->OnUpdate(m_TimeStep_secs);

		//Clear and Resize
		glClear(GL_COLOR_BUFFER_BIT);
		{
			int width, height;
			glfwGetWindowSize(m_WindowHandle, &(width), &(height));
			glViewport(0, 0, width, height);
		}

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		//Begin-------------------------------------------------------------------------------------------

		static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

		// We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
		// because it would be confusing to have two docking targets within each others.
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;

		const ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(viewport->WorkPos);
		ImGui::SetNextWindowSize(viewport->WorkSize);
		ImGui::SetNextWindowViewport(viewport->ID);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
		window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

		// When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
		// and handle the pass-thru hole, so we ask Begin() to not render a background.
		if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
			window_flags |= ImGuiWindowFlags_NoBackground;

		// Important: note that we proceed even if Begin() returns false (aka window is collapsed).
		// This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
		// all active windows docked into it will lose their parent and become undocked.
		// We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
		// any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		ImGui::Begin("DockSpace Demo", nullptr, window_flags);
		ImGui::PopStyleVar();

		ImGui::PopStyleVar(2);

		// Submit the DockSpace
		ImGuiIO& io = ImGui::GetIO();
		if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
		{
			ImGuiID dockspace_id = ImGui::GetID("GLDockspace");
			ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
		}

		//Render layers----------------------------------------------------------------------------------
		for (auto& layer : m_LayerStack)
			layer->OnUIRender();

		ImGui::End();

		//End---------------------------------------------------------------------------------------------
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			GLFWwindow* backup_current_context = glfwGetCurrentContext();
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
			glfwMakeContextCurrent(backup_current_context);
		}

		glfwSwapBuffers(m_WindowHandle);

		//capture delta time----------------------------------------------------------
		float time_secs = GetTime_seconds();
		m_FrameTime_secs = time_secs - m_LastFrameTime_secs;
		m_TimeStep_secs = glm::min<float>(m_FrameTime_secs, 0.0333f);
		m_LastFrameTime_secs = time_secs;
	}
}

Application::~Application()
{
	Shutdown();

	s_Instance = nullptr;
}

Application& Application::Get()
{
	return *s_Instance;
}

void Application::Close()
{
	m_Running = false;
}

//wrong unit
float Application::GetTime_seconds()
{
	return (float)glfwGetTime();
}

void Application::Init()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	m_WindowHandle = glfwCreateWindow(m_Specification.Width, m_Specification.Height, m_Specification.Name.c_str(), NULL, NULL);
	glfwMakeContextCurrent(m_WindowHandle);
	gladLoadGL();

	// imgui init stuff-------------
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

	ImGui::StyleColorsDark();

	//setup renderer backend
	ImGui_ImplGlfw_InitForOpenGL(m_WindowHandle, true);

	ImGui_ImplOpenGL3_Init(m_GLSL_Version);

	glViewport(0, 0, m_Specification.Width, m_Specification.Height);
	glClearColor(0.07f, 0.13f, 0.17f, 1.0f);//navy blue default window color
}

void Application::Shutdown()
{
	for (auto& layer : m_LayerStack)
		layer->OnDetach();

	m_LayerStack.clear();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(m_WindowHandle);
	glfwTerminate();
	g_ApplicationRunning = false;
}