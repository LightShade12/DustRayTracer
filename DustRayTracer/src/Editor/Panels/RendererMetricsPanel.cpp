#include "RendererMetricsPanel.hpp"

#include "Editor/Application/Application.hpp"
#include "Core/Renderer.hpp"
#include "Core/Scene/Camera.cuh"

#include <imgui.h>

void RendererMetricsPanel::OnUIRender(float last_frame_time_ms, float last_render_time_ms)
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
	ImGui::Text("%.3fms", Application::Get().GetFrameTimeSecs() * 1000);//???
	ImGui::TableSetColumnIndex(2);
	ImGui::Text("%d hz", int(1 / Application::Get().GetFrameTimeSecs()));

	ImGui::TableNextRow();
	ImGui::TableSetColumnIndex(0);
	ImGui::Text("GUI frame time(EditorLayer)");
	ImGui::TableSetColumnIndex(1);
	ImGui::Text("%.3fms", last_frame_time_ms);
	ImGui::TableSetColumnIndex(2);
	ImGui::Text("%d hz", int(1000 / last_frame_time_ms));

	ImGui::TableNextRow();
	ImGui::TableSetColumnIndex(0);
	ImGui::Text("CPU code execution time");
	ImGui::TableSetColumnIndex(1);
	ImGui::Text("%.3fms", (last_frame_time_ms - last_render_time_ms));
	ImGui::TableSetColumnIndex(2);
	ImGui::Text("%d hz", int(1000 / (last_frame_time_ms - last_render_time_ms)));

	//TODO: this is quite "regarded"
	//if moving/rendering
	if (m_Renderer->getSampleCount() < m_Renderer->m_RendererSettings.max_samples)
	{
		//if first frame/moving after render complete
		if (skip) { skip = false; renderfreqavg = 0; framecounter = 0; rendercumulation = 0; }
		else
		{
			framecounter++;
			renderfreq = 1000 / last_render_time_ms;
			renderfreqmin = fminf(renderfreq, renderfreqmin);
			rendercumulation += renderfreq;
			renderfreqavg = rendercumulation / framecounter;
			renderfreqmax = fmaxf(renderfreq, renderfreqmax);
		}
	}
	else if (m_Renderer->getSampleCount() == 0) { skip = true; printf("sample 0"); }
	else
	{//render complete/not moving; stationary display
		skip = true;
	}

	ImGui::TableNextRow();
	ImGui::TableSetColumnIndex(0);
	ImGui::Text("GPU Kernel time");
	ImGui::TableSetColumnIndex(1);
	ImGui::Text("%.3fms", last_render_time_ms);
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
	ImGui::Text("%d", m_Renderer->getSampleCount());

	ImGui::TableNextRow();
	ImGui::TableSetColumnIndex(0);
	ImGui::Text("View position");
	ImGui::TableSetColumnIndex(1);
	float3 pos = DeviceCamera->GetPosition();
	ImGui::Text("x: %.3f y: %.3f z: %.3f", pos.x, pos.y, pos.z);

	ImGui::TableNextRow();
	ImGui::TableSetColumnIndex(0);
	ImGui::Text("View direction");
	ImGui::TableSetColumnIndex(1);
	float3 fdir = DeviceCamera->m_Forward_dir;
	ImGui::Text("x: %.3f y: %.3f z: %.3f", fdir.x, fdir.y, fdir.z);

	ImGui::EndTable();

	ImGui::End();
}

RendererMetricsPanel::~RendererMetricsPanel()
{
}