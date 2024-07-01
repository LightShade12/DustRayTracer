#include "MaterialManagerPanel.hpp"
#include "Core/Scene/Scene.cuh"
#include "Editor/Common/CudaCommon.cuh"
#include <imgui.h>

void MaterialManagerPanel::Initialize(Scene* scene)
{
	m_Scene = scene;
}

bool MaterialManagerPanel::OnUIRender()
{
	bool refreshRender = false;
	ImGui::Begin("Material Manager");
	ImGui::Separator();
	static int selected_material_idx = 0;
	if (ImGui::BeginListBox("###Materials", ImVec2(ImGui::GetContentRegionAvail().x / 4, ImGui::GetContentRegionAvail().y)))
	{
		for (int n = 0; n < m_Scene->m_Materials.size(); n++)
		{
			const bool is_selected = (selected_material_idx == n);
			if (ImGui::Selectable(m_Scene->m_Materials[n].Name, is_selected)) {
				selected_material_idx = n;
				glDeleteTextures(1, &(albedothumbnail.gl_texture_name));
				glDeleteTextures(1, &(emissionthumbnail.gl_texture_name));
				glDeleteTextures(1, &(normalthumbnail.gl_texture_name));
				glDeleteTextures(1, &(roughnessthumbnail.gl_texture_name));
				if (m_Scene->m_Materials[selected_material_idx].AlbedoTextureIndex >= 0)
					albedothumbnail = makeThumbnail((m_Scene->m_Textures[m_Scene->m_Materials[selected_material_idx].AlbedoTextureIndex]));
				if (m_Scene->m_Materials[selected_material_idx].EmissionTextureIndex >= 0)
					emissionthumbnail = makeThumbnail((m_Scene->m_Textures[m_Scene->m_Materials[selected_material_idx].EmissionTextureIndex]));
				if (m_Scene->m_Materials[selected_material_idx].NormalTextureIndex >= 0)
					normalthumbnail = makeThumbnail((m_Scene->m_Textures[m_Scene->m_Materials[selected_material_idx].NormalTextureIndex]));
				if (m_Scene->m_Materials[selected_material_idx].RoughnessTextureIndex >= 0)
					roughnessthumbnail = makeThumbnail((m_Scene->m_Textures[m_Scene->m_Materials[selected_material_idx].RoughnessTextureIndex]));
			}

			// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
			if (is_selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndListBox();
	}
	ImGui::SameLine();
	ImGui::BeginChild("proppanel", ImGui::GetContentRegionAvail());
	ImGui::Text("Material Properties");
	ImGui::Separator();
	Material& selected_material = m_Scene->m_Materials[selected_material_idx];

	ImGui::Text(selected_material.Name);
	if (selected_material.AlbedoTextureIndex >= 0)
		ImGui::Image((void*)albedothumbnail.gl_texture_name, ImVec2(128 * (albedothumbnail.width / (float)albedothumbnail.height), 128));
	if (selected_material.EmissionTextureIndex >= 0) {
		ImGui::SameLine();
		ImGui::Image((void*)emissionthumbnail.gl_texture_name, ImVec2(128 * (emissionthumbnail.width / (float)emissionthumbnail.height), 128));
	}
	if (selected_material.NormalTextureIndex >= 0) {
		ImGui::SameLine();
		ImGui::Image((void*)normalthumbnail.gl_texture_name, ImVec2(128 * (normalthumbnail.width / (float)normalthumbnail.height), 128));
	}
	if (selected_material.RoughnessTextureIndex >= 0) {
		ImGui::SameLine();
		ImGui::Image((void*)roughnessthumbnail.gl_texture_name, ImVec2(128 * (roughnessthumbnail.width / (float)roughnessthumbnail.height), 128));
	}

	refreshRender |= ImGui::ColorEdit3("Albedo", &selected_material.Albedo.x);
	refreshRender |= ImGui::ColorEdit3("Emission color", &selected_material.EmissiveColor.x);
	refreshRender |= ImGui::SliderFloat("Emission scale", &selected_material.EmissiveScale, 0, 50);
	refreshRender |= ImGui::SliderFloat("Metallicity", &(selected_material.Metallicity), 0, 1);
	refreshRender |= ImGui::SliderFloat("Reflectance", &(selected_material.Reflectance), 0, 1);
	refreshRender |= ImGui::SliderFloat("Roughness", &(selected_material.Roughness), 0, 1);
	refreshRender |= ImGui::SliderFloat("Normal scale", &(selected_material.NormalMapScale), 0, 2);
	ImGui::EndChild();

	ImGui::End();

	return refreshRender;
}

MaterialManagerPanel::~MaterialManagerPanel()
{
	glDeleteTextures(1, &(albedothumbnail.gl_texture_name));
	glDeleteTextures(1, &(emissionthumbnail.gl_texture_name));
	glDeleteTextures(1, &(normalthumbnail.gl_texture_name));
	glDeleteTextures(1, &(roughnessthumbnail.gl_texture_name));
}

MaterialManagerPanel::DRTThumbnail MaterialManagerPanel::makeThumbnail(const Texture& drt_texture)
{
	GLuint tex_name;
	glGenTextures(1, &tex_name);
	glBindTexture(GL_TEXTURE_2D, tex_name);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	size_t imagebuffersize = sizeof(unsigned char) * drt_texture.width * drt_texture.height * drt_texture.componentCount;
	unsigned char* data = (unsigned char*)malloc(imagebuffersize);

	cudaDeviceSynchronize();
	cudaMemcpy(data, drt_texture.d_data, imagebuffersize, cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());

	if (drt_texture.componentCount == 4)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, drt_texture.width, drt_texture.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	else
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, drt_texture.width, drt_texture.height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

	glBindTexture(GL_TEXTURE_2D, 0);
	free(data);

	return DRTThumbnail(tex_name, drt_texture.width, drt_texture.height);
}