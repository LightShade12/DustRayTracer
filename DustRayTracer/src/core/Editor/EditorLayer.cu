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
#include <tiny_gltf.h>

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

bool loadModel(tinygltf::Model& model, const char* filename) {
	tinygltf::TinyGLTF loader;
	std::string err;
	std::string warn;

	bool res = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
	if (!warn.empty()) {
		std::cout << "WARN: " << warn << std::endl;
	}

	if (!err.empty()) {
		std::cout << "ERR: " << err << std::endl;
	}

	if (!res)
		std::cout << "Failed to load glTF: " << filename << std::endl;
	else
		std::cout << "Loaded glTF: " << filename << std::endl;

	return res;
}

__host__ void EditorLayer::OnAttach()
{
	m_dcamera = new Camera();
	m_Scene = new Scene();
	//------------------------------------------------------------------------

	tinygltf::Model loadedmodel;
	if (!loadModel(loadedmodel, "./src/models/cube.glb"))
	{
		std::cout << "model loading error\n";
		std::abort();
	}

	//mesh looping
	for (size_t meshIdx = 0; meshIdx < loadedmodel.meshes.size(); meshIdx++)
	{
		std::vector<float3> loadedmodelpositions;
		std::vector<float3>loadedmodelnormals;

		printf("processing mesh index: %d\n", meshIdx);

		for (size_t primIdx = 0; primIdx < loadedmodel.meshes[meshIdx].primitives.size(); primIdx++)
		{
			int pos_attrib_accesorIdx = loadedmodel.meshes[meshIdx].primitives[primIdx].attributes["POSITION"];
			int nrm_attrib_accesorIdx = loadedmodel.meshes[meshIdx].primitives[primIdx].attributes["NORMAL"];
			int indices_accesorIdx = loadedmodel.meshes[meshIdx].primitives[primIdx].indices;

			tinygltf::Accessor pos_accesor = loadedmodel.accessors[pos_attrib_accesorIdx];
			tinygltf::Accessor nrm_accesor = loadedmodel.accessors[nrm_attrib_accesorIdx];
			tinygltf::Accessor indices_accesor = loadedmodel.accessors[indices_accesorIdx];

			int pos_accesor_byte_offset = pos_accesor.byteOffset;//redundant
			int nrm_accesor_byte_offset = nrm_accesor.byteOffset;//redundant
			int indices_accesor_byte_offset = indices_accesor.byteOffset;//redundant

			tinygltf::BufferView pos_bufferview = loadedmodel.bufferViews[pos_accesor.bufferView];
			tinygltf::BufferView nrm_bufferview = loadedmodel.bufferViews[nrm_accesor.bufferView];
			tinygltf::BufferView indices_bufferview = loadedmodel.bufferViews[indices_accesor.bufferView];

			int pos_buffer_byte_offset = pos_bufferview.byteOffset;
			int nrm_buffer_byte_offset = nrm_bufferview.byteOffset;
			tinygltf::Buffer cube_buffer = loadedmodel.buffers[0];//should alawys be zero

			printf("normals accesor count: %d\n", nrm_accesor.count);
			printf("positions accesor count: %d\n", pos_accesor.count);
			printf("indices accesor count: %d\n", indices_accesor.count);

			unsigned short* indicesbuffer = (unsigned short*)(cube_buffer.data.data());
			float3* positions_buffer = (float3*)(cube_buffer.data.data() + pos_buffer_byte_offset);
			float3* normals_buffer = (float3*)(cube_buffer.data.data() + nrm_buffer_byte_offset);

			for (int i = (indices_bufferview.byteOffset / 2); i < (indices_bufferview.byteLength + indices_bufferview.byteOffset) / 2; i++)
			{
				loadedmodelpositions.push_back(positions_buffer[indicesbuffer[i]]);
				loadedmodelnormals.push_back(normals_buffer[indicesbuffer[i]]);
			}
		}

		printf("constructed positions count: %d \n", loadedmodelpositions.size());//should be 36 for cube
		printf("constructed normals count: %d \n", loadedmodelnormals.size());//should be 36 for cube

		if (loadedmodelpositions.size() == loadedmodelnormals.size())
		{
			bool stop = false;
			printf("positions:\n");
			for (size_t i = 0; i < loadedmodelpositions.size(); i++)
			{
				if (i > 3 && i < loadedmodelpositions.size() - 3)
				{
					if (!stop)
					{
						printf("...\n");
						stop = true;
					}
					continue;
				}
				float3 pos = loadedmodelpositions[i];
				printf("x:%.3f y:%.3f z:%.3f\n", pos.x, pos.y, pos.z);
			}
			stop = false;
			printf("normals:\n");
			for (size_t i = 0; i < loadedmodelnormals.size(); i++)
			{
				if (i > 3 && i < loadedmodelnormals.size() - 3)
				{
					if (!stop)
					{
						printf("...\n");
						stop = true;
					}
					continue;
				}
				float3 nrm = loadedmodelnormals[i];
				printf("x:%.3f y:%.3f z:%.3f\n", nrm.x, nrm.y, nrm.z);
			}
		}
		else
		{
			printf("positions-normals count mismatch!\n");
		}

		printf("constructing mesh\n");
		Mesh loadedmesh(loadedmodelpositions, loadedmodelnormals);
		printf("adding mesh\n");
		m_Scene->m_Meshes.push_back(loadedmesh);
		printf("success\n");
	}

	Material red;
	red.Albedo = { .7,0,0 };
	Material blue;
	blue.Albedo = make_float3(.7, .7, .7);

	m_Scene->m_Material.push_back(red);
	m_Scene->m_Material.push_back(blue);

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
	ImGui::GetMousePos();

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

//general purpose input handler
bool processInput(GLFWwindow* window, Camera* cam, float delta)
{
	bool has_moved = false;
	float sensitivity = 1;
	//has_moved = false;
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
			if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			{
				has_moved = true;
				velocity.z -= 1;
			}
			if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			{
				has_moved = true;
				velocity.z += 1;
			}

			//strafe
			if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			{
				has_moved = true;
				velocity.x -= 1;
			}
			if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			{
				has_moved = true;
				velocity.x += 1;
			}

			//UP/DOWN
			if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
			{
				has_moved = true;
				velocity.y -= 1;
			}
			if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
			{
				has_moved = true;
				velocity.y += 1;
			}

			//TODO: Input::GetMouseDeltaDegrees?
			double mouseX;
			double mouseY;
			glfwGetCursorPos(window, &mouseX, &mouseY);

			// Normalizes and shifts the coordinates of the cursor such that they begin in the middle of the screen
			// and then "transforms" them into degrees
			float rotX = sensitivity * (float)(mouseY - (height / 2)) / height;
			float rotY = sensitivity * (float)(mouseX - (width / 2)) / width;

			float sin_x = sin(-rotY);
			float cos_x = cos(-rotY);

			float sin_y = sin(-rotX);
			float cos_y = cos(-rotX);

			float4 mousedeltadegrees = { sin_x, cos_x, sin_y, cos_y };

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
	m_LastApplicationFrameTime = ts;
	if (processInput(Application::Get().GetWindowHandle(), (m_dcamera), ts))
		m_Renderer.resetAccumulationBuffer();
}

void EditorLayer::OnDetach()
{
	delete m_dcamera;
	cudaDeviceSynchronize();
	delete m_Scene;
}