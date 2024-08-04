#pragma once

#include <glm/glm.hpp>
#include <cstdint>

/*
TODO: Camera InputProcess() which calls rotate and translate and has mouseInput specific code
add aspect ratio
Roll out own host side basic vec3 types for CmaeraData
add matrix manip
*/
struct Ray;
namespace DustRayTracer {
	struct CameraData;

	class HostCamera
	{
	public:

		HostCamera();
		HostCamera& operator = (const HostCamera& other);

		explicit HostCamera(CameraData* device_camera_data);
		void cleanup();

		void OnUpdate(glm::vec3 velocity, float delta);
		void Rotate(glm::vec4 delta_degrees);

		void updateDevice();
		CameraData* getDeviceCamera() { return m_DeviceCameraData; };
		CameraData* getHostCamera() const { return m_HostCameraData; };//Please dont own this
		const char* getName();
		char* getNamePtr();

		void setExposure(float exposure);
		float getExposure() const;
		float* getExposurePtr();

		void setVerticalFOV(float vertical_rad);
		float getVerticalFOV() const;
		float* getVerticalFOVPtr();

		void setMovementSpeed(float speed);
		float getMovementSpeed() const;
		float* getMovementSpeedPtr();

		void setDefocusConeAngle(float radians);
		float getDefocusConeAngle() const;
		float* getDefocusConeAnglePtr();

		void setFocusDistance(float metres);
		float getFocusDistance() const;
		float* getFocusDistancePtr();

		void setPosition(glm::vec3 position);
		glm::vec3 getPosition() const;
		float* getPositionPtr();

		void setLookDir(glm::vec3 direction);
		glm::vec3 getLookDir() const;
		float* getLookDirPtr();

	private:
		CameraData* m_DeviceCameraData = nullptr;//device ptr
		CameraData* m_HostCameraData = nullptr;
	};
}