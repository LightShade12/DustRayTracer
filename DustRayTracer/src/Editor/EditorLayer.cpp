#include "EditorLayer.hpp"

#include "Application/private/Input.hpp"

#include "Theme/EditorTheme.hpp"
#include "Application/Application.hpp"
#include "Editor/Common/Timer.hpp"
#include "Editor/Importer.hpp"

#include "Core/Scene/HostScene.hpp"
//#include "Core/Scene/Camera.cuh"
//#include "Core/BVH/BVHBuilder.cuh"

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
	ConsoleLogs.push_back("-------------------console initialized--------------------");
	ConsoleLogs.push_back("GLFW 3.4");
	ConsoleLogs.push_back("CUDA 12.4");
	ConsoleLogs.push_back("OPENGL 4.6");

	//------------------------------------------------------------------------
	m_CurrentScene = std::make_shared<DustRayTracer::HostScene>();
	m_CurrentScene->initialize();
	Importer importer;
	importer.loadGLTF("../models/test/refract_test.glb", *m_CurrentScene);

	m_Renderer.initialize();

	BVHBuilder bvhbuilder;
	bvhbuilder.m_TargetLeafPrimitivesCount = 8;
	bvhbuilder.m_BinCount = 32;
	bvhbuilder.BuildIterative(*m_CurrentScene);

	m_Renderer.updateScene(*m_CurrentScene);

	m_ActiveCamera = m_Renderer.getCameraPtr();

	m_RendererMetricsPanel.SetRenderer(m_Renderer);
	m_RendererMetricsPanel.SetCamera(m_ActiveCamera);
	m_MaterialManagerPanel.Initialize(m_CurrentScene.get());

	m_RendererMetricsPanel.m_DevMetrics.m_ObjectsCount = m_CurrentScene->getMeshesBufferSize();
	m_RendererMetricsPanel.m_DevMetrics.m_TrianglesCount = m_CurrentScene->getTrianglesBufferSize();
	m_RendererMetricsPanel.m_DevMetrics.m_MaterialsCount = m_CurrentScene->getMaterialsBufferSize();
	m_RendererMetricsPanel.m_DevMetrics.m_TexturesCount = m_CurrentScene->getTexturesBufferSize();

	stbi_flip_vertically_on_write(true);

	//ImGuiThemes::modifiedDark();
	ImGuiThemes::RedOni();
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
			bool refreshrender = false;
			if (ImGui::BeginTabItem("Renderer"))
			{
				//static int renderer_mode = (int)m_Renderer.m_RendererSettings.RenderMode;
				//static int debug_view = (int)m_Renderer.m_RendererSettings.DebugMode;
				ImGui::Text("Renderer mode:"); ImGui::SameLine();
				refreshrender |= ImGui::Combo("###Renderer mode", (int*)&m_Renderer.m_RendererSettings.RenderMode, "Normal\0Debug");
				//{
				//	m_Renderer.m_RendererSettings.RenderMode = (RendererSettings::RenderModes)renderer_mode; m_Renderer.clearAccumulation();
				//}

				//NORMAL RENDER MODE MENU
				if ((RendererSettings::RenderModes)m_Renderer.m_RendererSettings.RenderMode == RendererSettings::RenderModes::NORMALMODE) {
					refreshrender |= (ImGui::Checkbox("Sunlight(ShadowRays)", &(m_Renderer.m_RendererSettings.enableSunlight)));
					refreshrender |= (ImGui::Checkbox("Gamma correction(2.0)", &(m_Renderer.m_RendererSettings.enable_gamma_correction)));
					refreshrender |= (ImGui::Checkbox("Enable MIS", &(m_Renderer.m_RendererSettings.useMIS)));
					refreshrender |= (ImGui::Checkbox("Invert Normal Map reading", &(m_Renderer.m_RendererSettings.invert_normal_map)));
					refreshrender |= (ImGui::Checkbox("Tone mapping", &(m_Renderer.m_RendererSettings.enable_tone_mapping)));
					ImGui::Text("Ray bounce limit:"); ImGui::SameLine();
					refreshrender |= (ImGui::InputInt("###Ray bounce limit:", &(m_Renderer.m_RendererSettings.ray_bounce_limit)));
					ImGui::Text("Max samples limit:"); ImGui::SameLine();
					refreshrender |= (ImGui::InputInt("###Max samples limit:", &(m_Renderer.m_RendererSettings.max_samples)));

					refreshrender |= (ImGui::Checkbox("Use Material Override: ", &(m_Renderer.m_RendererSettings.UseMaterialOverride)));

					if (m_Renderer.m_RendererSettings.UseMaterialOverride) {
						ImGui::Indent();
						refreshrender |= (ImGui::ColorEdit3("Global albedo: ", &(m_Renderer.m_RendererSettings.OverrideMaterial.Albedo.x)));
						refreshrender |= (ImGui::SliderFloat("Global metallic: ", &(m_Renderer.m_RendererSettings.OverrideMaterial.Metallicity), 0, 1));
						refreshrender |= (ImGui::SliderFloat("Global reflectance: ", &(m_Renderer.m_RendererSettings.OverrideMaterial.Reflectance), 0, 1));
						refreshrender |= (ImGui::SliderFloat("Global roughness: ", &(m_Renderer.m_RendererSettings.OverrideMaterial.Roughness), 0, 1));
						ImGui::Unindent();
						ImGui::Separator();
					}
				}
				else {//DEBUG MODE
					ImGui::Text("Debug view:"); ImGui::SameLine();
					refreshrender |= (ImGui::Combo("###Debug view", (int*)&m_Renderer.m_RendererSettings.DebugMode, "Albedo\0Normals\0Barycentric\0UVs\0BVH"));
					//{
					//	m_Renderer.m_RendererSettings.DebugMode = (RendererSettings::DebugModes)debug_view; m_Renderer.clearAccumulation();
					//}
				}
				ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_Leaf);
				//refreshrender |= (ImGui::InputFloat3("Position", (m_ActiveCamera->getPositionPtr())));
				//refreshrender |= (ImGui::InputFloat3("Direction", m_ActiveCamera->getLookDirPtr()));
				refreshrender |= (ImGui::SliderFloat("Speed", (m_ActiveCamera->getMovementSpeedPtr()), 0, 10, "%.3f", ImGuiSliderFlags_Logarithmic));
				refreshrender |= (ImGui::SliderAngle("Field-Of-View(Degrees)", m_ActiveCamera->getVerticalFOVPtr(), 5, 120));
				refreshrender |= (ImGui::SliderFloat("Focus Distance(m)", m_ActiveCamera->getFocusDistancePtr(), 0, 50, "%.3f", ImGuiSliderFlags_Logarithmic));
				refreshrender |= (ImGui::SliderFloat("Defocus Angle(Cone)", m_ActiveCamera->getDefocusConeAnglePtr(), 0, 2));
				refreshrender |= (ImGui::SliderFloat("Exposure", m_ActiveCamera->getExposurePtr(), 0, 10));
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Scene"))
			{
				ImGui::CollapsingHeader("Sun light", ImGuiTreeNodeFlags_Leaf);
				refreshrender |= (ImGui::ColorEdit3("Sunlight color", &(m_Renderer.m_RendererSettings.sunlight_color.x)));
				refreshrender |= (ImGui::SliderFloat("Sunlight size", &(m_Renderer.m_RendererSettings.sun_size), 0, 5));
				refreshrender |= (ImGui::SliderFloat("Sunlight intensity", &(m_Renderer.m_RendererSettings.sunlight_intensity), 0, 10));
				refreshrender |= (ImGui::SliderAngle("Sunlight Y rotation", &(m_Renderer.m_RendererSettings.sunlight_dir.x)));
				refreshrender |= (ImGui::SliderAngle("Sunlight altitude", &(m_Renderer.m_RendererSettings.sunlight_dir.y), 0, 90));
				ImGui::CollapsingHeader("Sky light", ImGuiTreeNodeFlags_Leaf);
				refreshrender |= (ImGui::ColorEdit3("Sky color", &(m_Renderer.m_RendererSettings.sky_color.x)));
				refreshrender |= (ImGui::SliderFloat("Sky intensity", &(m_Renderer.m_RendererSettings.sky_intensity), 0, 10));
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
			if (refreshrender) {
				m_Renderer.clearAccumulation();
				m_ActiveCamera->updateDevice();
				m_Renderer.updateRendererConfig(m_Renderer.m_RendererSettings);
			};
		}
		ImGui::End();
	}

	ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollbar);
	ImGui::SetNextWindowSize(ImVec2(640 + 13, 360 + 45));

	ImVec2 vpdims = ImGui::GetContentRegionAvail();

	if (m_Renderer.getRenderTargetImage_name() != NULL)
		ImGui::Image((void*)(uintptr_t)m_Renderer.getRenderTargetImage_name(),
			ImVec2(m_Renderer.getFrameWidth(), m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });

	ImGui::BeginChild("statusbar", ImVec2(ImGui::GetContentRegionAvail().x, 14), 0);

	//ImGui::SetCursorScreenPos({ ImGui::GetCursorScreenPos().x + 5, ImGui::GetCursorScreenPos().y + 4 });

	ImGui::Text("dims: %d x %d px", m_Renderer.getFrameWidth(), m_Renderer.getFrameHeight());
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
	static bool savesuccess = false;

	{
		Mesh selected_mesh = m_CurrentScene->getMesh(node_clicked);
		ImGui::Begin("Properties");
		if (ImGui::Button("Save frame as PNG"))
		{
			std::vector<GLubyte> frame_data(m_Renderer.getFrameWidth() * m_Renderer.getFrameHeight() * 4);//RGBA8
			glBindTexture(GL_TEXTURE_2D, m_Renderer.getRenderTargetImage_name());
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame_data.data());
			glBindTexture(GL_TEXTURE_2D, 0);

			savesuccess = saveImage("image", m_Renderer.getFrameWidth(), m_Renderer.getFrameHeight(), frame_data.data());
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

		for (int i = 0; i < m_CurrentScene->getMeshesBufferSize(); i++)
		{
			ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAvailWidth;
			// Disable the default "open on single-click behavior" + set Selected flag according to our selection.
			// To alter selection we use IsItemClicked() && !IsItemToggledOpen(), so clicking on an arrow doesn't alter selection.
			const bool is_selected = (selection_mask & (1 << i)) != 0;

			if (is_selected)
				node_flags |= ImGuiTreeNodeFlags_Selected;

			node_flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen; // ImGuiTreeNodeFlags_Bullet
			ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, m_CurrentScene->getMesh(i).Name);
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

	if (vpdims.y > 14)vpdims.y -= 12;//TODO: make this sensible var; not a constant
	if (vpdims.y < 5)vpdims.y = 10;
	m_Renderer.resizeResolution(uint32_t(vpdims.x), uint32_t(vpdims.y));
	m_Renderer.renderFrame(&m_LastRenderTime_ms);//make lastrendertime a member var of renderer and access it?

	m_LastFrameTime_ms = timer.ElapsedMillis();

	if (m_MaterialManagerPanel.OnUIRender())m_Renderer.clearAccumulation();
	m_RendererMetricsPanel.OnUIRender(m_LastFrameTime_ms, m_LastRenderTime_ms);
}
//TODO: make proper Cuda C++ interface and wrappers
//general purpose input handler
bool processInput(GLFWwindow* window, DustRayTracer::HostCamera* cam, float delta)
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

	if (processInput(Application::Get().GetWindowHandle(), m_ActiveCamera, ts)) {
		m_Renderer.clearAccumulation(); m_ActiveCamera->updateDevice();
	}
}

void EditorLayer::OnDetach()
{
	printf("detaching app\n");
	m_CurrentScene->Cleanup();
}