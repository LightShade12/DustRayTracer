#include "Camera.hpp"
#include "Core/Ray.cuh"
#include "Core/CudaMath/Random.cuh"
#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>
/*
camera(float vfov, glm::vec3 lookfrom, glm::vec3 lookdir, glm::vec3 vup, float aperture, float focus_dist) {
	auto theta = degrees_to_radians(vfov);
	auto fov_factor = tan(theta / 2);
	float viewport_height = 2.0 * fov_factor;
	float viewport_width = viewport_height;

	auto focal_length = 1.0;

	w = glm::normalize(lookdir);
	u = glm::normalize(cross(vup, w));
	v = cross(w, u);

	origin = lookfrom;
	horizontal = focus_dist * viewport_width * u;
	vertical = focus_dist * viewport_height * v;
	lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - focus_dist * w;

	lens_radius = aperture / 2;
}

ray get_ray(float s, float t) const {
	glm::vec3 rd = lens_radius * random_in_unit_disk();
	glm::vec3 offset = u * rd.x + v * rd.y;

	return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
}
public:
	glm::vec3 w;//front
	glm::vec3 u;//right
	glm::vec3 v;//up
	glm::vec3 origin;
	glm::vec3 lower_left_corner;
	glm::vec3 horizontal;
	glm::vec3 vertical;
*/
__host__ __device__ float deg2rad(float degree)
{
	float const PI = 3.14159265359f;
	return (degree * (PI / 180.f));
}

namespace DustRayTracer {
	HostCamera::HostCamera(CameraData* device_camera_data)
	{
		printf("downloading device camera\n");
		m_DeviceCameraData = device_camera_data;
		cudaMemcpy(&m_HostCameraData, m_DeviceCameraData, sizeof(CameraData), cudaMemcpyDeviceToHost);
	}
	//TODO: wdym OnUpdate hadles this??
	__host__ void HostCamera::OnUpdate(float3 velocity, float delta)
	{
		glm::mat3 cameramodelmatrix =
		{
			m_HostCameraData.right_vector.x,m_HostCameraData.right_vector.y,m_HostCameraData.right_vector.z,
			m_HostCameraData.upward_vector.x,m_HostCameraData.upward_vector.y,m_HostCameraData.upward_vector.z,
			m_HostCameraData.forward_vector.x,m_HostCameraData.forward_vector.y,m_HostCameraData.forward_vector.z
		};

		glm::vec3 vel = { velocity.x,velocity.y, velocity.z };
		glm::vec3 transformedVel = cameramodelmatrix * vel;
		float3 finalvel = { transformedVel.x, transformedVel.y,transformedVel.z };//not vel but movedir

		m_HostCameraData.position += m_HostCameraData.movement_speed * finalvel * delta;
	}

	//TODO: esoteric...
	__host__ void HostCamera::Rotate(float4 delta_degrees)
	{
		float sin_x = delta_degrees.x;
		float cos_x = delta_degrees.y;
		float sin_y = delta_degrees.z;
		float cos_y = delta_degrees.w;

		float3 rotated = (m_HostCameraData.forward_vector * cos_x) +
			(cross(m_HostCameraData.upward_vector, m_HostCameraData.forward_vector) * sin_x) +
			(m_HostCameraData.upward_vector * dot(m_HostCameraData.upward_vector, m_HostCameraData.forward_vector)) * (1 - cos_x);
		// Calculates upcoming vertical change in the Orientation
		m_HostCameraData.forward_vector = rotated;

		rotated = (m_HostCameraData.forward_vector * cos_y) +
			(cross(m_HostCameraData.right_vector, m_HostCameraData.forward_vector) * sin_y) +
			(m_HostCameraData.right_vector * dot(m_HostCameraData.right_vector, m_HostCameraData.forward_vector)) * (1 - cos_y);
		// Calculates upcoming vertical change in the Orientation
		m_HostCameraData.forward_vector = rotated;
		m_HostCameraData.right_vector = cross(m_HostCameraData.forward_vector, m_HostCameraData.upward_vector);
	}

	void HostCamera::updateDevice()
	{
		if (m_DeviceCameraData) {
			printf("updated camera data\n");
			cudaMemcpy(m_DeviceCameraData, &m_HostCameraData, sizeof(CameraData), cudaMemcpyHostToDevice);
		}
	}

	void HostCamera::setPosition(glm::vec3 position)
	{
		m_HostCameraData.position = make_float3(position.x, position.y, position.z);
	}

	void HostCamera::setLookDir(glm::vec3 direction)
	{
		m_HostCameraData.forward_vector = make_float3(direction.x, direction.y, direction.z);
	}

	//returns normalized dir
	__device__ Ray CameraData::getRay(float2 _uv, float width, float height, uint32_t& seed) const
	{
		float theta = vertical_fov_radians / 2;
		float fov_factor = tan(theta / 2.0f);

		float aspect_ratio = width / height;
		float world_image_plane_height = 2.0f * fov_factor * focus_dist;
		float world_image_plane_width = world_image_plane_height * aspect_ratio;

		float3 forward_dir = normalize(forward_vector);
		float3 right_dir = normalize(cross(forward_dir, make_float3(0, 1, 0)));
		float3 up_dir = cross(right_dir, forward_dir);

		float3 world_image_plane_horizontal_vector = world_image_plane_width * right_dir;
		float3 world_image_plane_vertical_vector = world_image_plane_height * up_dir;

		float2 offset = { randomFloat(seed) - 0.5f, randomFloat(seed) - 0.5f };
		offset *= 0.0035f; // Adjust scale as needed for anti-aliasing

		float defocus_radius = focus_dist * tan(deg2rad(defocus_cone_angle) / 2.0f);
		float3 defocus_disk_u = defocus_radius * right_dir;
		float3 defocus_disk_v = defocus_radius * up_dir;

		float3 rorig;

		if (defocus_cone_angle <= 0)
		{
			rorig = position;
		}
		else
		{
			float2 p = random_in_unit_disk(seed);
			rorig = position + (p.x * defocus_disk_u) + (p.y * defocus_disk_v);
		}

		float3 ray_direction = normalize((forward_dir * focus_dist) +
			((_uv.x + offset.x) * world_image_plane_horizontal_vector) +
			((_uv.y + offset.y) * world_image_plane_vertical_vector) -
			rorig + position);

		return Ray(rorig, ray_direction);
	}
}