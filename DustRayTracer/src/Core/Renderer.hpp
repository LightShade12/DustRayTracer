#pragma once
/*
* PathTracerRenderer is the high level render API that client interfaces for orchestrating the pathtracing
*/
#include "Scene/SceneData.cuh"
#include "Scene/HostScene.hpp"
#include "Scene/HostCamera.hpp"
#include <glad/glad.h>
#include <cstdint>

/*
* TODO:
* -BSDF
* -full pbr texture support
* -ASVGF
* -bloom
* -fog
* -projection
* -fsr
* -godrays
* -matrix lib
* -physical sky
* -profiler
* -instance geo
* -dynamic geometry
* -dynamic material api
* -compressed wide bvh
* -light bvh
*/

struct ThrustRGB32FBufferWrapper;//indirection to thrust device vector kernel code inclusion
struct CudaGLAPI;

namespace DustRayTracer {
	//should just be renderer?
	class PathTracerRenderer
	{
	public:
		PathTracerRenderer();
		~PathTracerRenderer();

		bool initialize();
		bool shutdown();

		void updateScene(DustRayTracer::HostScene& scene_object);
		void updateRendererConfig(const RendererSettings& config) { m_RendererSettings = config; m_DeviceSceneData.RenderSettings = m_RendererSettings; };
		//DustRayTracer::HostCamera getCamera() const;//client should get a copy of current camera to control
		DustRayTracer::HostCamera* getCameraPtr();//client should get a copy of current camera to control
		void changeCamera(uint32_t camera_idx);
		void resizeResolution(uint32_t frame_width, uint32_t frame_height);

		void renderFrame(float* frame_delta_time_ms);

		GLuint& getRenderTargetImage_name() { return m_RenderTargetTexture_name; };//TODO: should this be non const reference?
		GLuint& getColorBufferTargetImage_name() { return m_ColorBufferTargetTexture_name; };//should this be non const reference?
		GLuint& getNormalBufferTargetImage_name() { return m_NormalBufferTargetTexture_name; };//should this be non const reference?
		GLuint& getDepthBufferTargetImage_name() { return m_DepthBufferTargetTexture_name; }//should this be non const reference?

		uint32_t getFrameWidth() const { return m_BufferWidth; }
		uint32_t getFrameHeight() const { return m_BufferHeight; }

		//path tracer
		uint32_t getSampleIndex() const { return m_FrameIndex; }
		void clearAccumulation();
		//GLuint& getDirectDiffuseTargetImage_name();
		//GLuint& getIndirectDiffuseTargetImage_name();
		//GLuint& getDirectSpecularTargetImage_name();
		//GLuint& getIndirectSpecularTargetImage_name();

	public:
		RendererSettings m_RendererSettings;
	private:
		SceneData m_DeviceSceneData;
		uint32_t m_BufferWidth = 0, m_BufferHeight = 0;//specify as RenderBufferWidth/Height?
		GLuint m_RenderTargetTexture_name = NULL;//null init val important for triggering init
		GLuint m_ColorBufferTargetTexture_name = NULL;
		GLuint m_NormalBufferTargetTexture_name = NULL;
		GLuint m_DepthBufferTargetTexture_name = NULL;

		DustRayTracer::HostCamera m_CurrentCamera;

		CudaGLAPI* m_CudaGLAPI=nullptr;

		ThrustRGB32FBufferWrapper* m_AccumulationFrameBuffer;

		uint32_t m_FrameIndex = 1;

		int m_ThreadBlock_x = 8;
		int m_ThreadBlock_y = 8;
		dim3 m_BlockGridDimensions;
		dim3 m_ThreadBlockDimensions;
	};
}