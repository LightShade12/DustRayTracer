#include "EditorLayer.hpp"

#include "core/Application/Application.hpp"
#include "core/Common/Timer.hpp"
#include "core/Renderer/private/Shapes/Scene.cuh"//has thrust so editor needs to be compiled by nvcc
#include "core/Renderer/private/Camera/Camera.cuh"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>

#include <stb_image_write.h>

//appends extension automatically;png
bool saveImage(const char* filename, int _width, int _height, GLubyte* data)
{
	if (
		stbi_write_png((std::string(std::string(filename) + ".png")).c_str(), _width, _height, 3, data, 3 * sizeof(GLubyte) * _width)
		)
		return true;
	else
		return false;
}


__host__ void EditorLayer::OnAttach()
{
	m_dcamera = new Camera();
	m_Scene = new Scene();
	Sphere s1;
	s1.Position = { 1.5,0,-2.5 };
	s1.Albedo = { 0,1,0 };

	Sphere s2;
	s2.Position = { 0,0,0 };
	s2.Albedo = { 1,0,1 };

	Sphere s3;
	s3.Position = { -1.5,0,-2.5 };
	s3.Albedo = { 0,1,1 };
	
	Sphere s4;
	s4.Albedo = { 0,0,1 };
	s4.Position = {0,-101,0};
	s4.Radius=100;

	//m_Scene->m_Spheres.push_back(s1); //<-Problem line
	m_Scene->m_Spheres.push_back(s2); //<-Problem line
	//m_Scene->m_Spheres.push_back(s3); //<-Problem line
	m_Scene->m_Spheres.push_back(s4); //<-Problem line

	m_ObjectsCount = m_Scene->m_Spheres.size();

	stbi_flip_vertically_on_write(true);
}

void EditorLayer::OnUIRender()
{
	//-------------------------------------------------------------------------------------------------
	Timer timer;

	ImGui::Begin("test");
	ImGui::Text(msg.c_str());
	if (ImGui::Button("save png"))
	{
		std::vector<GLubyte> frame_data(m_Renderer.m_BufferWidth * m_Renderer.m_BufferHeight * 4);
		glBindTexture(GL_TEXTURE_2D, m_Renderer.GetRenderTargetImage_name());
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame_data.data());
		glBindTexture(GL_TEXTURE_2D, 0);

		if (saveImage("image", m_Renderer.m_BufferWidth, m_Renderer.m_BufferHeight, frame_data.data()))
			msg = "Image saved";
		else
			msg = "Image save failed";
	}
	ImGui::End();

	ImGui::Begin("Dev Metrics");
	ImGui::Text("GUI frame time(EditorLayer): %.3fms", m_LastFrameTime);
	ImGui::Text("Application frame time: %.3fms", Application::Get().GetFrameTime()*1000);
	ImGui::Text("CPU code execution time: %.3fms", (m_LastFrameTime-m_LastRenderTime));
	ImGui::Text("GPU Kernel time: %.3fms", m_LastRenderTime);
	ImGui::Text("Objects in scene: %d", m_ObjectsCount);

	ImGui::End();

	ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(100, 100));
	ImGui::Begin("Viewport");
	ImVec2 vpdims = ImGui::GetContentRegionAvail();
	if (m_Renderer.GetRenderTargetImage_name() != NULL)
		ImGui::Image((void*)(uintptr_t)m_Renderer.GetRenderTargetImage_name(),
			ImVec2(m_Renderer.m_BufferWidth, m_Renderer.m_BufferHeight), { 0,1 }, { 1,0 });
	ImGui::End();
	ImGui::PopStyleVar();

	ImGui::Begin("Padding window");
	ImGui::End();

	m_Renderer.ResizeBuffer(uint32_t(vpdims.x), uint32_t(vpdims.y));
	m_Renderer.Render(m_dcamera,*m_Scene, &m_LastRenderTime);//make lastrendertime a member var of renderer and access it?

	m_LastFrameTime = timer.ElapsedMillis();
}

//input handling
bool firstclick = true;
bool moving = false;
void processInput(GLFWwindow* window, Camera* cam, float delta)
{
	moving = false;
	//movement lateral
	float cameraSpeed = static_cast<float>(5 * delta);

	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		moving = true;
		(cam)->m_Position -= cameraSpeed * (cam)->m_Forward_dir;
	}
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		moving = true;
		float3 movedir = cameraSpeed * (cam)->m_Forward_dir;
		//printf("key pressed, x:%.3f, y:%.3f, z:%.3f\n", movedir.x,movedir.y,movedir.z);
		(cam)->m_Position += movedir;
	}

	//strafe
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		moving = true;
		(cam)->m_Position -= normalize((cam)->m_Right_dir) * cameraSpeed;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		moving = true;
		(cam)->m_Position += normalize((cam)->m_Right_dir) * cameraSpeed;
	}

	//UP/DOWN
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
	{
		moving = true;
		(cam)->m_Position -= normalize((cam)->m_Up_dir) * cameraSpeed;
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		moving = true;
		(cam)->m_Position += normalize((cam)->m_Up_dir) * cameraSpeed;
	}
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
		moving = true;
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
	processmouse(Application::Get().GetWindowHandle(), (m_dcamera), width, height);
	processInput(Application::Get().GetWindowHandle(), (m_dcamera), ts);
}

void EditorLayer::OnDetach()
{
	delete m_dcamera;
	delete m_Scene;
}