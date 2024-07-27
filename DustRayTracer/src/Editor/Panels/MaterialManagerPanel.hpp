#pragma once
#include <glad/glad.h>
#include <stdint.h>
namespace DustRayTracer {
	struct HostScene;
}

class Texture;

class MaterialManagerPanel
{
public:

	MaterialManagerPanel() = default;
	MaterialManagerPanel(DustRayTracer::HostScene* scene) :m_CurrentScene(scene) {};

	void Initialize(DustRayTracer::HostScene* scene);

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
	DRTThumbnail emissionthumbnail;
	DRTThumbnail normalthumbnail;
	DRTThumbnail roughnessthumbnail;
	DustRayTracer::HostScene* m_CurrentScene;//non owning
};