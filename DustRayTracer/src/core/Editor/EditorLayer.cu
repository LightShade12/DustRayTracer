#include "EditorLayer.hpp"

#include "Theme/EditorTheme.hpp"
#include "core/Application/Application.hpp"
#include "core/Common/Timer.hpp"
#include "core/Renderer/private/Shapes/Scene.cuh"//has thrust so editor needs to be compiled by nvcc
#include "core/Renderer/private/Camera/Camera.cuh"
#include "core/Renderer/private/CudaMath/helper_math.cuh"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>

#include <stb_image_write.h>

//appends extension automatically;png
bool saveImage(const char* filename, int _width, int _height, GLubyte* data)
{
	if (
		stbi_write_png((std::string(std::string(filename) + ".png")).c_str(), _width, _height, 4, data, 4 * sizeof(GLubyte) * _width)
		)
		return true;
	else
		return false;
}

__host__ void EditorLayer::OnAttach()
{
	m_dcamera = new Camera();
	m_Scene = new Scene();

	std::vector<float3>planefloorpositions =
	{ make_float3(2.5f, -0.2f, -2.5f), make_float3(2.5f, -0.2f, 2.5f), make_float3(-2.5f, -0.2f, -2.5f),
	make_float3(-2.5f, -0.2f, -2.5f), make_float3(2.5f, -0.2f, 2.5f), make_float3(-2.5f, -0.2f, 2.5f) };
	std::vector<float3>planefloornormals =
	{
		make_float3(0, 1, 0),
		make_float3(0, 1, 0)
	};
	std::vector<float3> cubepositions =
	{
		make_float3(0.5f, 1.5f, -0.5f),make_float3(0.5f, 1.5f, 0.5f), make_float3(-0.5f, 1.5f, -0.5f),
		make_float3(-0.5f, 1.5f, -0.5f), make_float3(0.5f, 1.5f, 0.5f), make_float3(-0.5f, 1.5f, 0.5f),
		make_float3(-0.5f, 0.5f, -0.5f), make_float3(0.5f, 0.5f, -0.5f),  make_float3(0.5f, 1.5f, -0.5f),
		make_float3(-0.5f, 0.5f, -0.5f), make_float3(0.5f, 1.5f, -0.5f), make_float3(-0.5f, 1.5f, -0.5f),
		make_float3(-0.5f, 0.5f, 0.5f), make_float3(0.5f, 0.5f, 0.5f), make_float3(0.5f, 1.5f, 0.5f),
		make_float3(-0.5f, 0.5f, 0.5f), make_float3(0.5f, 1.5f, 0.5f), make_float3(-0.5f, 1.5f, 0.5f),
		make_float3(-0.5f, 0.5f, -0.5f),make_float3(-0.5f, 0.5f, 0.5f),	make_float3(-0.5f, 1.5f, -0.5f),
		make_float3(-0.5f, 0.5f, 0.5f),make_float3(-0.5f, 1.5f, 0.5f),make_float3(-0.5f, 1.5f, -0.5f),
		make_float3(0.5f, 0.5f, -0.5f),make_float3(0.5f, 0.5f, 0.5f),make_float3(0.5f, 1.5f, -0.5f),
		make_float3(0.5f, 0.5f, 0.5f),make_float3(0.5f, 1.5f, 0.5f),make_float3(0.5f, 1.5f, -0.5f),
		make_float3(-0.5f, 0.5f, -0.5f),make_float3(0.5f, 0.5f, -0.5f),make_float3(0.5f, 0.5f, 0.5f),
		make_float3(-0.5f, 0.5f, -0.5f),make_float3(0.5f, 0.5f, 0.5f),make_float3(-0.5f, 0.5f, 0.5f)
	};
	std::vector<float3> cubenormals =
	{
		{ 0,1,0 }, { 0,1,0 }, { 0,0,-1 },
		{ 0,0,-1 }, { 0,0,1 }, { 0,0,1 },
		{ -1,0,0 }, { -1,0,0 }, { 1,0,0 },
		{ 1,0,0 },{ 0,-1,0 }, { 0,-1,0 }
	};

	Material red;
	red.Albedo = { .7,0,0 };
	Material blue;
	blue.Albedo = make_float3(.7, .7, .7);

	m_Scene->m_Material.push_back(red);
	m_Scene->m_Material.push_back(blue);

	Mesh planefloormesh(planefloorpositions, planefloornormals, 1);
	Mesh cubemesh(cubepositions, cubenormals);

	m_Scene->m_Meshes.push_back(planefloormesh);
	m_Scene->m_Meshes.push_back(cubemesh);

	m_DevMetrics.m_ObjectsCount = m_Scene->m_Meshes.size();

	for (Mesh mesh : m_Scene->m_Meshes)
	{
		m_DevMetrics.m_TrianglesCount += mesh.m_trisCount;
	}

	m_DevMetrics.m_MaterialsCount = m_Scene->m_Material.size();

	stbi_flip_vertically_on_write(true);

	ImGuithemes::dark();
	//ImGuithemes::UE4();
}

void EditorLayer::OnUIRender()
{
	//-------------------------------------------------------------------------------------------------
	Timer timer;

	//ImGui::ShowDemoWindow();

	{
		ImGui::Begin("test");
		ImGui::Text(msg.c_str());
		if (ImGui::Button("save png"))
		{
			std::vector<GLubyte> frame_data(m_Renderer.getBufferWidth() * m_Renderer.getBufferHeight() * 4);//RGBA8
			glBindTexture(GL_TEXTURE_2D, m_Renderer.GetRenderTargetImage_name());
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame_data.data());
			glBindTexture(GL_TEXTURE_2D, 0);

			if (saveImage("image", m_Renderer.getBufferWidth(), m_Renderer.getBufferHeight(), frame_data.data()))
				msg = "Image saved";
			else
				msg = "Image save failed";
		}
		ImGui::End();
	}

	{
		ImGui::Begin("Developer Metrics");

		ImGui::CollapsingHeader("Timers", ImGuiTreeNodeFlags_Leaf);

		ImGui::BeginTable("timerstable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg);
		ImGui::TableSetupColumn("Timer");
		ImGui::TableSetupColumn("Milli Seconds(ms)");
		ImGui::TableSetupColumn("Frequency(hz)");
		ImGui::TableHeadersRow();

		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("Application frame time");
		ImGui::TableSetColumnIndex(1);
		ImGui::Text("%.3fms", Application::Get().GetFrameTime() * 1000);
		ImGui::TableSetColumnIndex(2);
		ImGui::Text("%d hz", int(1 / Application::Get().GetFrameTime()));

		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("GUI frame time(EditorLayer)");
		ImGui::TableSetColumnIndex(1);
		ImGui::Text("%.3fms", m_LastFrameTime);
		ImGui::TableSetColumnIndex(2);
		ImGui::Text("%d hz", int(1000 / m_LastFrameTime));

		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("CPU code execution time");
		ImGui::TableSetColumnIndex(1);
		ImGui::Text("%.3fms", (m_LastFrameTime - m_LastRenderTime));
		ImGui::TableSetColumnIndex(2);
		ImGui::Text("%d hz", int(1000 / (m_LastFrameTime - m_LastRenderTime)));

		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("GPU Kernel time");
		ImGui::TableSetColumnIndex(1);
		ImGui::Text("%.3fms", m_LastRenderTime);
		ImGui::TableSetColumnIndex(2);
		ImGui::Text("%d hz", int(1000 / m_LastRenderTime));

		ImGui::EndTable();

		ImGui::CollapsingHeader("Geometry", ImGuiTreeNodeFlags_Leaf);

		ImGui::BeginTable("geometrytable", 2);

		ImGui::TableSetupColumn("Data");
		ImGui::TableSetupColumn("Value");
		ImGui::TableHeadersRow();

		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("Objects in scene");
		ImGui::TableSetColumnIndex(1);
		ImGui::Text("%d", m_DevMetrics.m_ObjectsCount);

		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("Net triangles in scene");
		ImGui::TableSetColumnIndex(1);
		ImGui::Text("%d", m_DevMetrics.m_TrianglesCount);

		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("Materials loaded");
		ImGui::TableSetColumnIndex(1);
		ImGui::Text("%d", m_DevMetrics.m_MaterialsCount);

		ImGui::EndTable();

		ImGui::CollapsingHeader("Renderer", ImGuiTreeNodeFlags_Leaf);

		ImGui::BeginTable("renderertable", 2);
		ImGui::TableSetupColumn("Property");
		ImGui::TableSetupColumn("Value");
		ImGui::TableHeadersRow();

		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("Samples");
		ImGui::TableSetColumnIndex(1);
		ImGui::Text("%d", m_Renderer.getSampleCount());

		ImGui::EndTable();

		ImGui::End();
	}

	ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(100, 100));
	ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollbar);
	ImVec2 vpdims = ImGui::GetContentRegionAvail();
	if (m_Renderer.GetRenderTargetImage_name() != NULL)
		ImGui::Image((void*)(uintptr_t)m_Renderer.GetRenderTargetImage_name(),
			ImVec2(m_Renderer.getBufferWidth(), m_Renderer.getBufferHeight()), { 0,1 }, { 1,0 });

	ImGui::BeginChild("statusbar", ImVec2(ImGui::GetContentRegionAvail().x, 14), 0);

	//ImGui::SetCursorScreenPos({ ImGui::GetCursorScreenPos().x + 5, ImGui::GetCursorScreenPos().y + 4 });

	ImGui::Text("dims: %d x %d px", m_Renderer.getBufferWidth(), m_Renderer.getBufferHeight());
	ImGui::SameLine();
	ImGui::Text(" | RGBA8");
	ImGui::EndChild();
	ImGui::End();
	ImGui::PopStyleVar(1);

	ImGui::Begin("Padding window");
	ImGui::End();

	vpdims.y -= 12;
	m_Renderer.ResizeBuffer(uint32_t(vpdims.x), uint32_t(vpdims.y));
	m_Renderer.Render(m_dcamera, (*m_Scene), &m_LastRenderTime);//make lastrendertime a member var of renderer and access it?

	m_LastFrameTime = timer.ElapsedMillis();
}

//input handling
bool firstclick = true;
bool moving = false;
void processInput(GLFWwindow* window, Camera* cam, float delta)
{
	moving = false;
	float3 velocity = { 0,0,0 };

	//movement lateral
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		moving = true;
		velocity -= (cam)->m_Forward_dir;
	}
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		moving = true;
		velocity += (cam)->m_Forward_dir;
	}

	//strafe
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		moving = true;
		velocity -= normalize((cam)->m_Right_dir);
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		moving = true;
		velocity += normalize((cam)->m_Right_dir);
	}

	//UP/DOWN
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
	{
		moving = true;
		velocity -= normalize((cam)->m_Up_dir);
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		moving = true;
		velocity += normalize((cam)->m_Up_dir);
	}

	cam->OnUpdate(velocity, delta);
}

void processmouse(GLFWwindow* window, Camera* cam, int width, int height)
{
	float sensitivity = 1;

	// Handles mouse inputs
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
	{
		// Hides mouse cursor
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		// Prevents camera from jumping on the first click
		if (firstclick)
		{
			glfwSetCursorPos(window, (width / 2), (height / 2));
			firstclick = false;
		}

		// Stores the coordinates of the cursor
		double mouseX;
		double mouseY;
		// Fetches the coordinates of the cursor
		glfwGetCursorPos(window, &mouseX, &mouseY);

		// Normalizes and shifts the coordinates of the cursor such that they begin in the middle of the screen
		// and then "transforms" them into degrees
		float rotX = sensitivity * (float)(mouseY - (height / 2)) / height;
		float rotY = sensitivity * (float)(mouseX - (width / 2)) / width;

		//printf("rotx: %.3f, toty: %.3f\n",rotX,rotY);

		float sin_x = sin(-rotY);
		float cos_x = cos(-rotY);
		float3 rotated = (cam->m_Forward_dir * cos_x) +
			(cross(cam->m_Up_dir, cam->m_Forward_dir) * sin_x) +
			(cam->m_Up_dir * dot(cam->m_Up_dir, cam->m_Forward_dir)) * (1 - cos_x);
		// Calculates upcoming vertical change in the Orientation
		//printf("x: %.3f, y:%.3f, z:%.3f\n", rotated.x, rotated.y, rotated.z);
		cam->m_Forward_dir = rotated;

		float sin_y = sin(-rotX);
		float cos_y = cos(-rotX);
		rotated = (cam->m_Forward_dir * cos_y) +
			(cross(cam->m_Right_dir, cam->m_Forward_dir) * sin_y) +
			(cam->m_Right_dir * dot(cam->m_Right_dir, cam->m_Forward_dir)) * (1 - cos_y);
		// Calculates upcoming vertical change in the Orientation
		cam->m_Forward_dir = rotated;
		cam->m_Right_dir = cross(cam->m_Forward_dir, cam->m_Up_dir);
		rotated = cam->m_Position;

		//printf("x: %.3f, y:%.3f, z:%.3f\n", rotated.x, rotated.y, rotated.z);

		//lookdir = glm::rotate(lookdir, -rotY, cam->m_Up_dir);//yaw
		//lookdir = glm::rotate(lookdir, -rotX, cam->m_Right_dir);//pitch

		// Sets mouse cursor to the middle of the screen so that it doesn't end up roaming around
		glfwSetCursorPos(window, (width / 2), (height / 2));
		if (rotX != 0 || rotY != 0)moving = true;
	}

	else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE)
	{
		// Unhides cursor since camera is not looking around anymore
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		// Makes sure the next time the camera looks around it doesn't jump
		firstclick = true;
	}
}

void EditorLayer::OnUpdate(float ts)
{
	int width = 0, height = 0;
	m_LastApplicationFrameTime = ts;
	glfwGetWindowSize(Application::Get().GetWindowHandle(), &width, &height);

	processInput(Application::Get().GetWindowHandle(), (m_dcamera), ts);//resets bool moving so comes first
	processmouse(Application::Get().GetWindowHandle(), (m_dcamera), width, height);
	if (moving) m_Renderer.resetAccumulationBuffer();
}

void EditorLayer::OnDetach()
{
	delete m_dcamera;
	cudaDeviceSynchronize();
	delete m_Scene;
}