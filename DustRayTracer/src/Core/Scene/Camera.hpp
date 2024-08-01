#pragma once
//#include "Core/CudaMath/helper_math.cuh"

#include <glm/glm.hpp>
#include <vector_types.h>
#include <cstdint>

/*
TODO: Camera InputProcess() which calls rotate and translate and has mouseInput specific code
add aspect ratio
Roll out own host side basic vec3 types for CmaeraData
add matrix manip
*/

struct Ray;

__host__ __device__ float deg2rad(float degree);

namespace DustRayTracer {
	struct CameraData {
		CameraData() = default;
		char name[32] = "unnamed";
		float exposure = 1.5;
		float vertical_fov_radians = deg2rad(60);
		float zfar = 0;
		float znear = 0;
		float aspect_ratio = 0;
		float defocus_cone_angle = 0;  // Variation angle of rays through each pixel
		float focus_dist = 10;

		float movement_speed = 10;

		//TODO: switch to glm
		float3 position = { 0,0,0 };
		float3 forward_vector = { 0,0,-1 };
		float3 upward_vector = { 0,1,0 };
		float3 right_vector = { 1,0,0 };
		uint32_t viewheight = 0, viewwidth = 0;//weird stuff
		__device__ Ray getRay(float2 screen_uv, float framewidth, float frameheight, uint32_t& seed) const;
	};

	class HostCamera
	{
	public:
		HostCamera() = default;
		explicit HostCamera(CameraData* device_camera_data);

		void OnUpdate(float3 velocity, float delta);
		void Rotate(float4 delta_degrees);

		void updateDevice();
		CameraData* getDeviceCamera() { return m_DeviceCameraData; };
		CameraData getHostCamera() const { return m_HostCameraData; };
		const char* getName() { return m_HostCameraData.name; };
		char* getNamePtr() { return m_HostCameraData.name; };

		void setExposure(float exposure) { m_HostCameraData.exposure = exposure; };
		float getExposure() const { return m_HostCameraData.exposure; };
		float* getExposurePtr() { return &(m_HostCameraData.exposure); };

		void setVerticalFOV(float vertical_rad) { m_HostCameraData.vertical_fov_radians = vertical_rad; };
		float getVerticalFOV() const { return m_HostCameraData.vertical_fov_radians; };
		float* getVerticalFOVPtr() { return &(m_HostCameraData.vertical_fov_radians); };

		void setMovementSpeed(float speed) { m_HostCameraData.movement_speed = speed; };
		float getMovementSpeed()const { return m_HostCameraData.movement_speed; };
		float* getMovementSpeedPtr() { return &(m_HostCameraData.movement_speed); };

		void setDefocusConeAngle(float radians) { m_HostCameraData.defocus_cone_angle = radians; };
		float getDefocusConeAngle()const { return m_HostCameraData.defocus_cone_angle; };
		float* getDefocusConeAnglePtr() { return &(m_HostCameraData.defocus_cone_angle); };

		void setFocusDistance(float metres) { m_HostCameraData.focus_dist = metres; };
		float getFocusDistance() const { return m_HostCameraData.focus_dist; };
		float* getFocusDistancePtr() { return &(m_HostCameraData.focus_dist); };

		void setPosition(glm::vec3 position);
		glm::vec3 getPosition() const { return glm::vec3(m_HostCameraData.position.x, m_HostCameraData.position.y, m_HostCameraData.position.z); };
		float* getPositionPtr() { return &(m_HostCameraData.position.x); };

		void setLookDir(glm::vec3 direction);
		glm::vec3 getLookDir() const { return glm::vec3(m_HostCameraData.forward_vector.x, m_HostCameraData.forward_vector.y, m_HostCameraData.forward_vector.z); };
		float* getLookDirPtr() { return &(m_HostCameraData.forward_vector.x); };

	private:
		CameraData* m_DeviceCameraData = nullptr;
		CameraData m_HostCameraData;
	};
}