#pragma once
#include "Core/Scene/Texture.cuh"

#include <glad/glad.h>
#include <stdint.h>

class Scene;

class MaterialManagerPanel
{
public:

	MaterialManagerPanel() = default;
	MaterialManagerPanel(Scene* scene) :m_Scene(scene) {};

	void Initialize(Scene* scene);

	bool OnUIRender();

	~MaterialManagerPanel();

private:
	struct DRTThumbnail
	{
		DRTThumbnail() = default;
		DRTThumbnail(GLuint name, uint32_t width_, uint32_t height_) :gl_texture_name(name), width(width_), height(height_) {};
		GLuint gl_texture_name = 0;
		uint32_t width = 0, height = 0;
		//TODO: texture is leaked on app close
	};
	DRTThumbnail makeThumbnail(const Texture& drt_texture);

private:
	//resources:
	DRTThumbnail albedothumbnail;
	Scene* m_Scene;//non owning
};