#include "EditorLayer.hpp"

#include "Application/private/Input.hpp"

#include "Theme/EditorTheme.hpp"
#include "Application/Application.hpp"
#include "Editor/Common/Timer.hpp"

#include "Core/Scene/Scene.cuh"
#include "Core/Scene/Camera.cuh"
#include "Core/BVH/BVHBuilder.cuh"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>

#include <stb_image_write.h>



/*
TODO:add denoiser settings
*/

//appends extension automatically;png
//input handling
bool firstclick = true;

bool EditorLayer::saveImage(const char* filename, int _width, int _height, GLubyte* data)
{
	return stbi_write_png((std::string(std::string(filename) + ".png")).c_str(),
		_width, _height, 4, data, 4 * sizeof(GLubyte) * _width);
}

void EditorLayer::OnAttach()
{
	//m_Renderer.m_RendererSettings.sky_intensity = 0;
	//m_device_Camera = new Camera(make_float3(-4.65, 1.8, -5));
	//m_device_Camera = new Camera(make_float3(0, 1, 2.8));
	//m_device_Camera = new Camera(make_float3(1.04, .175, .05));
	m_Scene = std::make_shared<Scene>();

	ConsoleLogs.push_back("-------------------console initialized--------------------");
	ConsoleLogs.push_back("GLFW 3.4");
	ConsoleLogs.push_back("CUDA 12.4");
	ConsoleLogs.push_back("OPENGL 4.6");

	//------------------------------------------------------------------------
	//m_Scene->loadGLTFmodel("../models/minecraft/mc_fort.glb", &m_device_Camera);
	m_Scene->loadGLTFmodel("../models/source/cs16_dust.glb", &m_device_Camera);
	//m_Scene->loadGLTFmodel("../models/test/emissive_test.glb", &m_device_Camera);
	//m_Scene->loadGLTFmodel("../models/test/cornell_box_v2.glb", &m_device_Camera);
	if (m_device_Camera == nullptr) { m_device_Camera = new Camera(make_float3(0, 1, 2.8)); }
	//m_device_Camera->m_Forward_dir = { .038,-.583,-.810 };
	m_device_Camera->m_movement_speed = 10.0;
	m_device_Camera->defocus_angle = 0.f;
	m_device_Camera->focus_dist = 10.f;
	m_device_Camera->exposure = 1.5f;

	m_RendererMetricsPanel.SetRenderer(m_Renderer);
	m_RendererMetricsPanel.SetCamera(m_device_Camera);
	m_MaterialManagerPanel.Initialize(m_Scene.get());

	BVHBuilder bvhbuilder;
	bvhbuilder.m_TargetLeafPrimitivesCount = 8;
	bvhbuilder.m_BinCount = 32;
	m_Scene->d_BVHTreeRoot = bvhbuilder.BuildIterative(m_Scene->m_PrimitivesBuffer, m_Scene->m_BVHNodes);

	for (size_t i = 0; i < m_Scene->m_PrimitivesBuffer.size(); i++)
	{
		const Triangle* tri = &m_Scene->m_PrimitivesBuffer[i];
		int mtid = tri->material_idx;
		if (m_Scene->m_Materials[mtid].EmissionTextureIndex >= 0 ||
			!(m_Scene->m_Materials[mtid].EmissiveColor.x == 0 &&
				m_Scene->m_Materials[mtid].EmissiveColor.y == 0 &&
				m_Scene->m_Materials[mtid].EmissiveColor.z == 0)) {
			m_Scene->m_MeshLights.push_back(i);
		}
	}
	printf("mesh lights %zu \n", m_Scene->m_MeshLights.size());
	//printToConsole("bvhtreeroot prims %zu\n", m_Scene->d_BVHTreeRoot->primitives_count);

	m_RendererMetricsPanel.m_DevMetrics.m_ObjectsCount = m_Scene->m_Meshes.size();

	for (Mesh mesh : m_Scene->m_Meshes)
	{
		m_RendererMetricsPanel.m_DevMetrics.m_TrianglesCount += mesh.m_trisCount;
	}

	m_RendererMetricsPanel.m_DevMetrics.m_MaterialsCount = m_Scene->m_Materials.size();
	m_RendererMetricsPanel.m_DevMetrics.m_TexturesCount = m_Scene->m_Textures.size();

	stbi_flip_vertically_on_write(true);

	ImGuiThemes::modifiedDark();
	//ImGuiThemes::OldSteam();
}

void EditorLayer::OnUIRender()
{
	//-------------------------------------------------------------------------------------------------
	Timer timer;

	ImGui::ShowDemoWindow();

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
					m_Renderer.m_RendererSettings.RenderMode = (RendererSettings::RenderModes)renderer_mode; m_Renderer.clearAccumulation();
				}

				if ((RendererSettings::RenderModes)renderer_mode == RendererSettings::RenderModes::NORMALMODE) {
					if (ImGui::Checkbox("Sunlight(ShadowRays)", &(m_Renderer.m_RendererSettings.enableSunlight)))m_Renderer.clearAccumulation();
					if (ImGui::Checkbox("Gamma correction(2.0)", &(m_Renderer.m_RendererSettings.enable_gamma_correction)))m_Renderer.clearAccumulation();
					if (ImGui::Checkbox("Enable MIS", &(m_Renderer.m_RendererSettings.useMIS)))m_Renderer.clearAccumulation();
					if (ImGui::Checkbox("Invert Normal Map reading", &(m_Renderer.m_RendererSettings.invert_normal_map)))m_Renderer.clearAccumulation();
					if (ImGui::Checkbox("Tone mapping", &(m_Renderer.m_RendererSettings.enable_tone_mapping)))m_Renderer.clearAccumulation();
					ImGui::Text("Ray bounce limit:"); ImGui::SameLine();
					if (ImGui::InputInt("###Ray bounce limit:", &(m_Renderer.m_RendererSettings.ray_bounce_limit)))m_Renderer.clearAccumulation();
					ImGui::Text("Max samples limit:"); ImGui::SameLine();
					if (ImGui::InputInt("###Max samples limit:", &(m_Renderer.m_RendererSettings.max_samples)))m_Renderer.clearAccumulation();

					if (ImGui::Checkbox("Use Material Override: ", &(m_Renderer.m_RendererSettings.UseMaterialOverride)))m_Renderer.clearAccumulation();
					if (m_Renderer.m_RendererSettings.UseMaterialOverride) {
						ImGui::Indent();
						if (ImGui::ColorEdit3("Global albedo: ", &(m_Renderer.m_RendererSettings.OverrideMaterial.Albedo.x)))m_Renderer.clearAccumulation();
						if (ImGui::SliderFloat("Global metallic: ", &(m_Renderer.m_RendererSettings.OverrideMaterial.Metallicity), 0, 1))m_Renderer.clearAccumulation();
						if (ImGui::SliderFloat("Global reflectance: ", &(m_Renderer.m_RendererSettings.OverrideMaterial.Reflectance), 0, 1))m_Renderer.clearAccumulation();
						if (ImGui::SliderFloat("Global roughness: ", &(m_Renderer.m_RendererSettings.OverrideMaterial.Roughness), 0, 1))m_Renderer.clearAccumulation();
						ImGui::Unindent();
						ImGui::Separator();
					}
				}
				else {//DEBUG MODE
					ImGui::Text("Debug view:"); ImGui::SameLine();
					if (ImGui::Combo("###Debug view", &debug_view, "Albedo\0Normals\0Barycentric\0UVs\0BVH"))
					{
						m_Renderer.m_RendererSettings.DebugMode = (RendererSettings::DebugModes)debug_view; m_Renderer.clearAccumulation();
					}
				}
				ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_Leaf);
				if (ImGui::InputFloat3("Position", &(m_device_Camera->m_Position.x)))m_Renderer.clearAccumulation();
				if (ImGui::InputFloat3("Direction", &(m_device_Camera->m_Forward_dir.x)))m_Renderer.clearAccumulation();
				if (ImGui::SliderFloat("Speed", &(m_device_Camera->m_movement_speed), 0, 10, "%.3f", ImGuiSliderFlags_Logarithmic))m_Renderer.clearAccumulation();
				if (ImGui::SliderAngle("Field-Of-View(Degrees)", &(m_device_Camera->vfov_rad), 5, 120))m_Renderer.clearAccumulation();
				if (ImGui::SliderFloat("Focus Distance(m)", &(m_device_Camera->focus_dist), 0, 50, "%.3f", ImGuiSliderFlags_Logarithmic))m_Renderer.clearAccumulation();
				if (ImGui::SliderFloat("Defocus Angle(Cone)", &(m_device_Camera->defocus_angle), 0, 2))m_Renderer.clearAccumulation();
				if (ImGui::SliderFloat("Exposure", &(m_device_Camera->exposure), 0, 10))m_Renderer.clearAccumulation();
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Scene"))
			{
				ImGui::CollapsingHeader("Sun light", ImGuiTreeNodeFlags_Leaf);
				if (ImGui::ColorEdit3("Sunlight color", (float*)&(m_Renderer.m_RendererSettings.sunlight_color)))m_Renderer.clearAccumulation();
				if (ImGui::SliderFloat("Sunlight size", &(m_Renderer.m_RendererSettings.sun_size), 0, 5))m_Renderer.clearAccumulation();
				if (ImGui::SliderFloat("Sunlight intensity", &(m_Renderer.m_RendererSettings.sunlight_intensity), 0, 10))m_Renderer.clearAccumulation();
				if (ImGui::SliderAngle("Sunlight Y rotation", &(m_Renderer.m_RendererSettings.sunlight_dir.x)))m_Renderer.clearAccumulation();
				if (ImGui::SliderAngle("Sunlight altitude", &(m_Renderer.m_RendererSettings.sunlight_dir.y), 0, 90))m_Renderer.clearAccumulation();
				ImGui::CollapsingHeader("Sky light", ImGuiTreeNodeFlags_Leaf);
				if (ImGui::ColorEdit3("Sky color", (float*)&(m_Renderer.m_RendererSettings.sky_color)))m_Renderer.clearAccumulation();
				if (ImGui::SliderFloat("Sky intensity", &(m_Renderer.m_RendererSettings.sky_intensity), 0, 10))m_Renderer.clearAccumulation();
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
		}

		ImGui::End();
	}

	ImGui::SetNextWindowSize(ImVec2(960 + 13, 500 + 45), ImGuiCond_Once);
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

	static int selection_mask = (1 << 2);
	static int node_clicked = 0;
	Mesh& selected_mesh = m_Scene->m_Meshes[node_clicked];
	static bool savesuccess = false;

	{
		ImGui::Begin("Properties");
		if (ImGui::Button("Save frame as PNG"))
		{
			std::vector<GLubyte> frame_data(m_Renderer.getBufferWidth() * m_Renderer.getBufferHeight() * 4);//RGBA8
			glBindTexture(GL_TEXTURE_2D, m_Renderer.GetRenderTargetImage_name());
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame_data.data());
			glBindTexture(GL_TEXTURE_2D, 0);

			savesuccess = saveImage("image", m_Renderer.getBufferWidth(), m_Renderer.getBufferHeight(), frame_data.data());
			ImGui::OpenPopup("savemsg");
		}
		ImGui::Separator();
		ImGui::CollapsingHeader("Mesh properties", ImGuiTreeNodeFlags_Leaf);
		ImGui::Text(selected_mesh.Name);
		ImGui::Text("Triangle count:%zu", selected_mesh.m_trisCount);
		ImGui::Separator();
		if (ImGui::BeginPopup("savemsg")) {
			if (savesuccess) ImGui::Text("Image saved");
			else ImGui::Text("Image saved failed");
			ImGui::EndPopup();
		}
		ImGui::End();
	}

	ImGui::Begin("SceneGraph");
	if (ImGui::TreeNodeEx("Root", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth)) {
		// 'selection_mask' is dumb representation of what may be user-side selection state.
	   //  You may retain selection state inside or outside your objects in whatever format you see fit.
	   // 'node_clicked' is temporary storage of what node we have clicked to process selection at the end
	   /// of the loop. May be a pointer to your own node type, etc.

		for (int i = 0; i < m_Scene->m_Meshes.size(); i++)
		{
			ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAvailWidth;
			// Disable the default "open on single-click behavior" + set Selected flag according to our selection.
			// To alter selection we use IsItemClicked() && !IsItemToggledOpen(), so clicking on an arrow doesn't alter selection.
			const bool is_selected = (selection_mask & (1 << i)) != 0;

			if (is_selected)
				node_flags |= ImGuiTreeNodeFlags_Selected;

			node_flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen; // ImGuiTreeNodeFlags_Bullet
			ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, m_Scene->m_Meshes[i].Name);
			if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
				node_clicked = i;
		}

		if (node_clicked != -1)
		{
			// Update selection state
			// (process outside of tree loop to avoid visual inconsistencies during the clicking frame)
			//if (ImGui::GetIO().KeyCtrl)
			//	selection_mask ^= (1 << node_clicked);          // CTRL+click to toggle
			//else //if (!(selection_mask & (1 << node_clicked))) // Depending on selection behavior you want, may want to preserve selection when clicking on item that is part of the selection
			selection_mask = (1 << node_clicked);           // Click to single-select
		}
		ImGui::Indent(ImGui::GetTreeNodeToLabelSpacing());
		ImGui::TreePop();
	};
	ImGui::End();

	if (vpdims.x > 14)vpdims.y -= 12;//TODO: make this sensible var; not a constant
	m_Renderer.ResizeBuffer(uint32_t(vpdims.x), uint32_t(vpdims.y));
	m_Renderer.Render(m_device_Camera, (*m_Scene), &m_LastRenderTime_ms);//make lastrendertime a member var of renderer and access it?

	m_LastFrameTime_ms = timer.ElapsedMillis();

	if (m_MaterialManagerPanel.OnUIRender())m_Renderer.clearAccumulation();
	m_RendererMetricsPanel.OnUIRender(m_LastFrameTime_ms, m_LastRenderTime_ms);
}
//TODO: make proper Cuda C++ interface and wrappers
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
		m_Renderer.clearAccumulation();
}

void EditorLayer::OnDetach()
{
	delete m_device_Camera;
}