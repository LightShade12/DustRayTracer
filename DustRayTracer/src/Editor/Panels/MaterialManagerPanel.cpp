#include "MaterialManagerPanel.hpp"
#include "Core/Public/DustRayTracer.hpp"
#include <imgui.h>

void MaterialManagerPanel::Initialize(DustRayTracer::HostScene* scene)
{
	m_CurrentScene = scene;
}

bool MaterialManagerPanel::OnUIRender()
{
	bool refreshRender = false;
	ImGui::Begin("Material Manager");
	ImGui::Separator();
	static int selected_material_idx = 0;
	if (ImGui::BeginListBox("###Materials", ImVec2(ImGui::GetContentRegionAvail().x / 4, ImGui::GetContentRegionAvail().y)))
	{
		for (int n = 0; n < m_CurrentScene->getMaterialsBufferSize(); n++)
		{
			const bool is_selected = (selected_material_idx == n);
			DustRayTracer::HostMaterial material = m_CurrentScene->getMaterial(n);
			if (ImGui::Selectable(material.getName(), is_selected)) {
				selected_material_idx = n;
				glDeleteTextures(1, &(albedothumbnail.gl_texture_name));
				glDeleteTextures(1, &(emissionthumbnail.gl_texture_name));
				glDeleteTextures(1, &(normalthumbnail.gl_texture_name));
				glDeleteTextures(1, &(roughnessthumbnail.gl_texture_name));
				if (material.getAlbedoTextureIndex() >= 0)
					albedothumbnail = makeThumbnail((m_CurrentScene->getTexture(material.getAlbedoTextureIndex())));
				if (material.getEmissionTextureIndex() >= 0)
					emissionthumbnail = makeThumbnail((m_CurrentScene->getTexture(material.getEmissionTextureIndex())));
				if (material.getNormalTextureIndex() >= 0)
					normalthumbnail = makeThumbnail((m_CurrentScene->getTexture(material.getNormalTextureIndex())));
				if (material.getRoughnessTextureIndex() >= 0)
					roughnessthumbnail = makeThumbnail((m_CurrentScene->getTexture(material.getRoughnessTextureIndex())));
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
	DustRayTracer::HostMaterial selected_material = m_CurrentScene->getMaterial(selected_material_idx);

	ImGui::Text(selected_material.getName());
	if (selected_material.getAlbedoTextureIndex() >= 0)
		ImGui::Image((void*)albedothumbnail.gl_texture_name, ImVec2(128 * (albedothumbnail.width / (float)albedothumbnail.height), 128));
	if (selected_material.getEmissionTextureIndex() >= 0) {
		ImGui::SameLine();
		ImGui::Image((void*)emissionthumbnail.gl_texture_name, ImVec2(128 * (emissionthumbnail.width / (float)emissionthumbnail.height), 128));
	}
	if (selected_material.getNormalTextureIndex() >= 0) {
		ImGui::SameLine();
		ImGui::Image((void*)normalthumbnail.gl_texture_name, ImVec2(128 * (normalthumbnail.width / (float)normalthumbnail.height), 128));
	}
	if (selected_material.getRoughnessTextureIndex() >= 0) {
		ImGui::SameLine();
		ImGui::Image((void*)roughnessthumbnail.gl_texture_name, ImVec2(128 * (roughnessthumbnail.width / (float)roughnessthumbnail.height), 128));
	}

	refreshRender |= ImGui::ColorEdit3("Albedo", selected_material.getAlbedoPtr());
	refreshRender |= ImGui::ColorEdit3("Emission color", selected_material.getEmissiveColorPtr());
	refreshRender |= ImGui::SliderFloat("Emission scale", selected_material.getEmissiveScalePtr(), 0, 50);
	refreshRender |= ImGui::SliderFloat("Metallicity", selected_material.getMetallicityPtr(), 0, 1);
	refreshRender |= ImGui::SliderFloat("Reflectance", selected_material.getReflectancePtr(), 0, 1);
	refreshRender |= ImGui::SliderFloat("Transmission", selected_material.getTransmissionPtr(), 0, 1);
	refreshRender |= ImGui::SliderFloat("IOR", selected_material.getIORPtr(), 1, 2);
	refreshRender |= ImGui::SliderFloat("Roughness", selected_material.getRoughnessPtr(), 0, 1);
	refreshRender |= ImGui::SliderFloat("Normal scale", selected_material.getNormalMapScalePtr(), 0, 2);

	if (refreshRender) selected_material.updateDevice();
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

	unsigned char* data = nullptr;
	drt_texture.getPixelsData(&data);

	if (drt_texture.componentCount == 4)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, drt_texture.width, drt_texture.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	else
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, drt_texture.width, drt_texture.height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

	glBindTexture(GL_TEXTURE_2D, 0);
	free(data);

	return DRTThumbnail(tex_name, drt_texture.width, drt_texture.height);
}