#include "EditorLayer.hpp"

#include "Application/private/Input.hpp"
#include "Common/dbg_macros.hpp"

#include "Theme/EditorTheme.hpp"
#include "Application/Application.hpp"
#include "Common/Timer.hpp"
#include "Core/Scene/Scene.cuh"//has thrust so editor needs to be compiled by nvcc
#include "Core/Scene/Camera.cuh"
#include "Core/CudaMath/helper_math.cuh"
#include "Core/BVH/BVHNode.cuh"
#include "Core/BVH/BVHBuilder.cuh"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>

#include <stb_image_write.h>

//appends extension automatically;png
//input handling
bool firstclick = true;

bool EditorLayer::saveImage(const char* filename, int _width, int _height, GLubyte* data)
{
	if (
		stbi_write_png((std::string(std::string(filename) + ".png")).c_str(), _width, _height, 4, data, 4 * sizeof(GLubyte) * _width)
		)
		return true;
	else
		return false;
}

void EditorLayer::OnAttach()
{
	m_device_Camera = new Camera(make_float3(6, 2.5, -36));
	//m_device_Camera = new Camera(make_float3(0, 2, 5));
	//m_device_Camera = new Camera(make_float3(1.04, .175, .05));
	m_device_Camera->m_movement_speed = 10.0;
	m_device_Camera->defocus_angle = 0.f;
	m_device_Camera->focus_dist = 10.f;
	//m_device_Camera->m_Forward_dir = { -1,0,0 };
	m_Scene = new Scene();

	ConsoleLogs.push_back("-------------------console initialized-------------------");
	ConsoleLogs.push_back("GLFW 3.4");
	ConsoleLogs.push_back("CUDA 12.4");
	ConsoleLogs.push_back("OPENGL 4.6");

	//------------------------------------------------------------------------
	m_Scene->loadGLTFmodel("../models/source/cs16_dust.glb");
	//m_Scene->loadGLTFmodel("../models/test/EmissiveTest.glb");

	BVHBuilder bvhbuilder;
	bvhbuilder.m_TargetLeafPrimitivesCount = 6;
	bvhbuilder.m_BinCount = 8;
	m_Scene->d_BVHTreeRoot = bvhbuilder.build(m_Scene->m_PrimitivesBuffer, m_Scene->m_BVHNodes);

	//printToConsole("bvhtreeroot prims %zu\n", m_Scene->d_BVHTreeRoot->primitives_count);

	m_DevMetrics.m_ObjectsCount = m_Scene->m_Meshes.size();

	for (Mesh mesh : m_Scene->m_Meshes)
	{
		m_DevMetrics.m_TrianglesCount += mesh.m_trisCount;
	}

	m_DevMetrics.m_MaterialsCount = m_Scene->m_Material.size();
	m_DevMetrics.m_TexturesCount = m_Scene->m_Textures.size();

	stbi_flip_vertically_on_write(true);

	ImGuiThemes::dark();
	//ImGuiThemes::UE4();
}

void EditorLayer::OnUIRender()
{
	//-------------------------------------------------------------------------------------------------
	Timer timer;

	//ImGui::ShowDemoWindow();

	{
		ImGui::Begin("test");
		ImGui::Text(test_window_text.c_str());
		if (ImGui::Button("save png"))
		{
			std::vector<GLubyte> frame_data(m_Renderer.getBufferWidth() * m_Renderer.getBufferHeight() * 4);//RGBA8
			glBindTexture(GL_TEXTURE_2D, m_Renderer.GetRenderTargetImage_name());
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame_data.data());
			glBindTexture(GL_TEXTURE_2D, 0);

			if (saveImage("image", m_Renderer.getBufferWidth(), m_Renderer.getBufferHeight(), frame_data.data()))
				test_window_text = "Image saved";
			else
				test_window_text = "Image save failed";
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
		ImGui::Text("%.3fms", Application::Get().GetFrameTime_secs() * 1000);//???
		ImGui::TableSetColumnIndex(2);
		ImGui::Text("%d hz", int(1 / Application::Get().GetFrameTime_secs()));

		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("GUI frame time(EditorLayer)");
		ImGui::TableSetColumnIndex(1);
		ImGui::Text("%.3fms", m_LastFrameTime_ms);
		ImGui::TableSetColumnIndex(2);
		ImGui::Text("%d hz", int(1000 / m_LastFrameTime_ms));

		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("CPU code execution time");
		ImGui::TableSetColumnIndex(1);
		ImGui::Text("%.3fms", (m_LastFrameTime_ms - m_LastRenderTime_ms));
		ImGui::TableSetColumnIndex(2);
		ImGui::Text("%d hz", int(1000 / (m_LastFrameTime_ms - m_LastRenderTime_ms)));

		//TODO: this is quite "regarded"
		//if moving/rendering
		if (m_Renderer.getSampleCount() < m_Renderer.m_RendererSettings.max_samples)
		{
			//if first frame/moving after render complete
			if (skip) { skip = false; renderfreqavg = 0; framecounter = 0; rendercumulation = 0; }
			else
			{
				framecounter++;
				renderfreq = 1000 / m_LastRenderTime_ms;
				renderfreqmin = fminf(renderfreq, renderfreqmin);
				rendercumulation += renderfreq;
				renderfreqavg = rendercumulation / framecounter;
				renderfreqmax = fmaxf(renderfreq, renderfreqmax);
			}
		}
		else if (m_Renderer.getSampleCount() == 0) { skip = true; printf("sample 0"); }
		else
		{//render complete/not moving; stationary display
			skip = true;
		}

		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("GPU Kernel time");
		ImGui::TableSetColumnIndex(1);
		ImGui::Text("%.3fms", m_LastRenderTime_ms);
		ImGui::TableSetColumnIndex(2);
		ImGui::Text("%.3f hz | (%.1f|%.1f|%.1f)", renderfreq, renderfreqmin, renderfreqavg, renderfreqmax);

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

		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("Textures loaded");
		ImGui::TableSetColumnIndex(1);
		ImGui::Text("%d", m_DevMetrics.m_TexturesCount);

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

		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("View position");
		ImGui::TableSetColumnIndex(1);
		float3 pos = m_device_Camera->GetPosition();
		ImGui::Text("x: %.3f y: %.3f z: %.3f", pos.x, pos.y, pos.z);

		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("View direction");
		ImGui::TableSetColumnIndex(1);
		float3 fdir = m_device_Camera->m_Forward_dir;
		ImGui::Text("x: %.3f y: %.3f z: %.3f", fdir.x, fdir.y, fdir.z);

		ImGui::EndTable();

		ImGui::End();
	}

	{
		ImGui::Begin("Settings");

		ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
		if (ImGui::BeginTabBar("MyTabBar", tab_bar_flags))
		{
			if (ImGui::BeginTabItem("Renderer"))
			{
				static int renderer_mode = (int)m_Renderer.m_RendererSettings.RenderMode;
				static int debug_view = (int)m_Renderer.m_RendererSettings.DebugMode;
				ImGui::Text("Renderer mode:"); ImGui::SameLine();
				if (ImGui::Combo("###Renderer mode", &renderer_mode, "Normal\0Debug")) {
					m_Renderer.m_RendererSettings.RenderMode = (RendererSettings::RenderModes)renderer_mode; m_Renderer.resetAccumulationBuffer();
				}

				if ((RendererSettings::RenderModes)renderer_mode == RendererSettings::RenderModes::NORMALMODE) {
					if (ImGui::Checkbox("Sunlight(ShadowRays)", &(m_Renderer.m_RendererSettings.enableSunlight)))m_Renderer.resetAccumulationBuffer();
					if (ImGui::Checkbox("Gamma correction(2.0)", &(m_Renderer.m_RendererSettings.gamma_correction)))m_Renderer.resetAccumulationBuffer();
					ImGui::Text("Ray bounce limit:"); ImGui::SameLine();
					if (ImGui::InputInt("###Ray bounce limit:", &(m_Renderer.m_RendererSettings.ray_bounce_limit)))m_Renderer.resetAccumulationBuffer();
					ImGui::Text("Max samples limit:"); ImGui::SameLine();
					if (ImGui::InputInt("###Max samples limit:", &(m_Renderer.m_RendererSettings.max_samples)))m_Renderer.resetAccumulationBuffer();
				}
				else {//DEBUG MODE
					ImGui::Text("Debug view:"); ImGui::SameLine();
					if (ImGui::Combo("###Debug view", &debug_view, "Albedo\0Normals\0Barycentric\0UVs\0MeshBVH\0WorldBounds"))
					{
						m_Renderer.m_RendererSettings.DebugMode = (RendererSettings::DebugModes)debug_view; m_Renderer.resetAccumulationBuffer();
					}
				}
				ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_Leaf);
				if (ImGui::SliderAngle("Field-Of-View(Degrees)", &(m_device_Camera->vfov_rad), 5, 120))m_Renderer.resetAccumulationBuffer();
				if (ImGui::SliderFloat("Focus Distance", &(m_device_Camera->focus_dist), 0, 50))m_Renderer.resetAccumulationBuffer();
				if (ImGui::SliderFloat("Defocus Angle(Cone)", &(m_device_Camera->defocus_angle), 0, 2))m_Renderer.resetAccumulationBuffer();
				if (ImGui::SliderFloat("Exposure", &(m_device_Camera->exposure), 0, 10))m_Renderer.resetAccumulationBuffer();

				ImGui::CollapsingHeader("Sun light", ImGuiTreeNodeFlags_Leaf);
				if (ImGui::ColorEdit3("Sunlight color", (float*)&(m_Renderer.m_RendererSettings.sunlight_color)))m_Renderer.resetAccumulationBuffer();
				if (ImGui::SliderFloat("Sunlight intensity", &(m_Renderer.m_RendererSettings.sunlight_intensity), 0, 10))m_Renderer.resetAccumulationBuffer();
				if (ImGui::SliderAngle("Sunlight Y rotation", &(m_Renderer.m_RendererSettings.sunlight_dir.x)))m_Renderer.resetAccumulationBuffer();
				if (ImGui::SliderAngle("Sunlight altitude", &(m_Renderer.m_RendererSettings.sunlight_dir.y), 0, 90))m_Renderer.resetAccumulationBuffer();
				ImGui::CollapsingHeader("Sky light", ImGuiTreeNodeFlags_Leaf);
				if (ImGui::ColorEdit3("Sky color", (float*)&(m_Renderer.m_RendererSettings.sky_color)))m_Renderer.resetAccumulationBuffer();
				if (ImGui::SliderFloat("Sky intensity", &(m_Renderer.m_RendererSettings.sky_intensity), 0, 10))m_Renderer.resetAccumulationBuffer();
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
		}

		ImGui::End();
	}

	//ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(100, 100));

	ImGui::SetNextWindowSize(ImVec2(860 + 16, 480 + 47));
	ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollbar);

	ImVec2 vpdims = ImGui::GetContentRegionAvail();

	if (m_Renderer.GetRenderTargetImage_name() != NULL)
		ImGui::Image((void*)(uintptr_t)m_Renderer.GetRenderTargetImage_name(),
			ImVec2(m_Renderer.getBufferWidth(), m_Renderer.getBufferHeight()), { 0,1 }, { 1,0 });

	ImGui::BeginChild("statusbar", ImVec2(ImGui::GetContentRegionAvail().x, 14), 0);

	//ImGui::SetCursorScreenPos({ ImGui::GetCursorScreenPos().x + 5, ImGui::GetCursorScreenPos().y + 4 });

	ImGui::Text("dims: %d x %d px", m_Renderer.getBufferWidth(), m_Renderer.getBufferHeight());
	ImGui::SameLine();
	ImGui::Text(" | RGBA32F");
	ImGui::EndChild();
	ImGui::End();
	//	ImGui::PopStyleVar(1);

	ImGui::Begin("Console window");
	for (const char* log : ConsoleLogs)
		ImGui::Text(log);
	ImGui::End();

	ImGui::Begin("SceneGraph");
	ImGui::End();

	vpdims.y -= 12;//TODO: make this sensible; not a constant
	m_Renderer.ResizeBuffer(uint32_t(vpdims.x), uint32_t(vpdims.y));
	m_Renderer.Render(m_device_Camera, (*m_Scene), &m_LastRenderTime_ms);//make lastrendertime a member var of renderer and access it?

	m_LastFrameTime_ms = timer.ElapsedMillis();
}

//general purpose input handler
bool processInput(GLFWwindow* window, Camera* cam, float delta)
{
	bool has_moved = false;
	float sensitivity = 1;
	float3 velocity = { 0,0,0 };
	int width = 0, height = 0;
	glfwGetWindowSize(window, &width, &height);

	//camera controls
	{
		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
		{
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

			// Prevents camera from jumping on the first click
			if (firstclick)
			{
				glfwSetCursorPos(window, (width / 2), (height / 2));
				firstclick = false;
			}

			//movement lateral
			if (Input::IsKeyDown(KeyCode::S))
			{
				has_moved = true;
				velocity.z -= 1;
			}
			if (Input::IsKeyDown(KeyCode::W))
			{
				has_moved = true;
				velocity.z += 1;
			}

			//strafe
			if (Input::IsKeyDown(KeyCode::A))
			{
				has_moved = true;
				velocity.x -= 1;
			}
			if (Input::IsKeyDown(KeyCode::D))
			{
				has_moved = true;
				velocity.x += 1;
			}

			//UP/DOWN
			if (Input::IsKeyDown(KeyCode::Q))
			{
				has_moved = true;
				velocity.y -= 1;
			}
			if (Input::IsKeyDown(KeyCode::E))
			{
				has_moved = true;
				velocity.y += 1;
			}

			//TODO: Input::GetMouseDeltaDegrees?
			glm::vec2 mousePos = Input::getMousePosition();
			glfwSetCursorPos(window, (width / 2), (height / 2));

			// Normalizes and shifts the coordinates of the cursor such that they begin in the middle of the screen
			// and then "transforms" them into degrees
			float rotX = sensitivity * (float)(mousePos.y - (height / 2)) / height;
			float rotY = sensitivity * (float)(mousePos.x - (width / 2)) / width;

			float sin_x = sin(-rotY);
			float cos_x = cos(-rotY);

			float sin_y = sin(-rotX);
			float cos_y = cos(-rotX);

			float4 mousedeltadegrees = { sin_x, cos_x, sin_y, cos_y };

			//printf("delta: %.5f\n", delta);

			cam->OnUpdate(velocity, delta);
			cam->Rotate(mousedeltadegrees);

			// Sets mouse cursor to the middle of the screen so that it doesn't end up roaming around
			glfwSetCursorPos(window, (width / 2), (height / 2));
			if (rotX != 0 || rotY != 0)has_moved = true;
		}
		//free mouse
		else
		{
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			// Makes sure the next time the camera looks around it doesn't jump
			firstclick = true;
		}
	}

	return has_moved;
}

void EditorLayer::OnUpdate(float ts)
{
	//m_LastApplicationFrameTime = ts;

	if (processInput(Application::Get().GetWindowHandle(), m_device_Camera, ts))
		m_Renderer.resetAccumulationBuffer();
}

void EditorLayer::OnDetach()
{
	delete m_device_Camera;
	cudaDeviceSynchronize();
	delete m_Scene;
}