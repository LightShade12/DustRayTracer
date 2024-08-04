#include <cuda.h>
#define GLM_FORCE_CUDA
#include "HostCamera.hpp"
#include "CameraData.cuh"
#include "Core/Ray.cuh"
#include "Core/CudaMath/Random.cuh"
#include <glm/mat3x3.hpp>
#include <vector_types.h>

namespace DustRayTracer {
	HostCamera::HostCamera()
	{
		m_HostCameraData = new CameraData;
	}

	HostCamera::HostCamera(CameraData* device_camera_data)
	{
		printf("downloading device camera\n");
		m_DeviceCameraData = device_camera_data;
		if (m_HostCameraData == nullptr)m_HostCameraData = new CameraData;
		cudaMemcpy(m_HostCameraData, m_DeviceCameraData, sizeof(CameraData), cudaMemcpyDeviceToHost);
	}

	HostCamera& HostCamera::operator=(const HostCamera& other)
	{
		if (this == &other) return *this;

		this->cleanup();
		this->m_HostCameraData = other.m_HostCameraData;
		this->m_DeviceCameraData = other.m_DeviceCameraData;

		return *this;
	}

	void HostCamera::cleanup()
	{
		if (m_HostCameraData != nullptr) {
			delete m_HostCameraData;
			m_HostCameraData = nullptr;
			printf("\ndeleted OldHostCameraData\n");
		}
	}
	//TODO: wdym OnUpdate hadles this??
	__host__ void HostCamera::OnUpdate(glm::vec3 velocity, float delta)
	{
		glm::mat3 cameramodelmatrix =
		{
			m_HostCameraData->right_vector.x,m_HostCameraData->right_vector.y,m_HostCameraData->right_vector.z,
			m_HostCameraData->upward_vector.x,m_HostCameraData->upward_vector.y,m_HostCameraData->upward_vector.z,
			m_HostCameraData->forward_vector.x,m_HostCameraData->forward_vector.y,m_HostCameraData->forward_vector.z
		};

		glm::vec3 vel = { velocity.x,velocity.y, velocity.z };
		glm::vec3 transformedVel = cameramodelmatrix * vel;
		float3 finalvel = { transformedVel.x, transformedVel.y,transformedVel.z };//not vel but movedir

		m_HostCameraData->position += m_HostCameraData->movement_speed * finalvel * delta;
	}

	//TODO: esoteric...
	__host__ void HostCamera::Rotate(glm::vec4 delta_degrees)
	{
		float sin_x = delta_degrees.x;
		float cos_x = delta_degrees.y;
		float sin_y = delta_degrees.z;
		float cos_y = delta_degrees.w;

		float3 rotated = (m_HostCameraData->forward_vector * cos_x) +
			(cross(m_HostCameraData->upward_vector, m_HostCameraData->forward_vector) * sin_x) +
			(m_HostCameraData->upward_vector * dot(m_HostCameraData->upward_vector, m_HostCameraData->forward_vector)) * (1 - cos_x);
		// Calculates upcoming vertical change in the Orientation
		m_HostCameraData->forward_vector = rotated;

		rotated = (m_HostCameraData->forward_vector * cos_y) +
			(cross(m_HostCameraData->right_vector, m_HostCameraData->forward_vector) * sin_y) +
			(m_HostCameraData->right_vector * dot(m_HostCameraData->right_vector, m_HostCameraData->forward_vector)) * (1 - cos_y);
		// Calculates upcoming vertical change in the Orientation
		m_HostCameraData->forward_vector = rotated;
		m_HostCameraData->right_vector = cross(m_HostCameraData->forward_vector, m_HostCameraData->upward_vector);
	}

	void HostCamera::updateDevice()
	{
		if (m_DeviceCameraData) {
			printf("updated camera data\n");
			cudaMemcpy(m_DeviceCameraData, m_HostCameraData, sizeof(CameraData), cudaMemcpyHostToDevice);
		}
	}

	const char* HostCamera::getName()
	{
		return m_HostCameraData->name;
	}

	char* HostCamera::getNamePtr() {
		return m_HostCameraData->name;
	}

	void HostCamera::setExposure(float exposure) {
		m_HostCameraData->exposure = exposure;
	}

	float HostCamera::getExposure() const {
		return m_HostCameraData->exposure;
	}

	float* HostCamera::getExposurePtr() {
		return &(m_HostCameraData->exposure);
	}

	void HostCamera::setVerticalFOV(float vertical_rad) {
		m_HostCameraData->vertical_fov_radians = vertical_rad;
	}

	float HostCamera::getVerticalFOV() const {
		return m_HostCameraData->vertical_fov_radians;
	}

	float* HostCamera::getVerticalFOVPtr() {
		return &(m_HostCameraData->vertical_fov_radians);
	}

	void HostCamera::setMovementSpeed(float speed) {
		m_HostCameraData->movement_speed = speed;
	}

	float HostCamera::getMovementSpeed() const {
		return m_HostCameraData->movement_speed;
	}

	float* HostCamera::getMovementSpeedPtr() {
		return &(m_HostCameraData->movement_speed);
	}

	void HostCamera::setDefocusConeAngle(float radians) {
		m_HostCameraData->defocus_cone_angle = radians;
	}

	float HostCamera::getDefocusConeAngle() const {
		return m_HostCameraData->defocus_cone_angle;
	}

	float* HostCamera::getDefocusConeAnglePtr() {
		return &(m_HostCameraData->defocus_cone_angle);
	}

	void HostCamera::setFocusDistance(float metres) {
		m_HostCameraData->focus_dist = metres;
	}

	float HostCamera::getFocusDistance() const {
		return m_HostCameraData->focus_dist;
	}

	float* HostCamera::getFocusDistancePtr() {
		return &(m_HostCameraData->focus_dist);
	}

	void HostCamera::setPosition(glm::vec3 position) {
		m_HostCameraData->position = make_float3(position.x, position.y, position.z);
	}

	glm::vec3 HostCamera::getPosition() const {
		return glm::vec3(m_HostCameraData->position.x, m_HostCameraData->position.y, m_HostCameraData->position.z);
	}

	float* HostCamera::getPositionPtr() {
		return &(m_HostCameraData->position.x);
	}

	void HostCamera::setLookDir(glm::vec3 direction) {
		m_HostCameraData->forward_vector = make_float3(direction.x, direction.y, direction.z);
	}

	glm::vec3 HostCamera::getLookDir() const {
		return glm::vec3(m_HostCameraData->forward_vector.x, m_HostCameraData->forward_vector.y, m_HostCameraData->forward_vector.z);
	}

	float* HostCamera::getLookDirPtr() {
		return &(m_HostCameraData->forward_vector.x);
	}
}