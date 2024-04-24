#include "Camera.cuh"

#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>

__device__ float deg2rad(float degree)
{
	float const PI = 3.14159265359;
	return (degree * (PI / 180));
}

/*
camera(float vfov, glm::vec3 lookfrom, glm::vec3 lookdir, glm::vec3 vup, float aperture, float focus_dist) {
	auto theta = degrees_to_radians(vfov);
	auto h = tan(theta / 2);
	float viewport_height = 2.0 * h;
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

__host__ void Camera::OnUpdate(float3 velocity, float delta)
{
	glm::mat3 cameramodelmatrix =
	{
		m_Right_dir.x,m_Right_dir.y,m_Right_dir.z,
		m_Up_dir.x,m_Up_dir.y,m_Up_dir.z,
		m_Forward_dir.x,m_Forward_dir.y,m_Forward_dir.z
	};

	glm::vec3 vel = { velocity.x,velocity.y, velocity.z };
	glm::vec3 transformedVel = cameramodelmatrix * vel;
	float3 finalvel = { transformedVel.x, transformedVel.y,transformedVel.z };

	m_Position += m_movement_speed * finalvel * delta;
}

__host__ void Camera::Rotate(float2 mouse_delta_degrees)
{
	float rotX = mouse_delta_degrees.x;
	float rotY = mouse_delta_degrees.y;

	float sin_x = sin(-rotY);
	float cos_x = cos(-rotY);
	float3 rotated = (m_Forward_dir * cos_x) +
		(cross(m_Up_dir, m_Forward_dir) * sin_x) +
		(m_Up_dir * dot(m_Up_dir, m_Forward_dir)) * (1 - cos_x);
	// Calculates upcoming vertical change in the Orientation
	m_Forward_dir = rotated;

	float sin_y = sin(-rotX);
	float cos_y = cos(-rotX);
	rotated = (m_Forward_dir * cos_y) +
		(cross(m_Right_dir, m_Forward_dir) * sin_y) +
		(m_Right_dir * dot(m_Right_dir, m_Forward_dir)) * (1 - cos_y);
	// Calculates upcoming vertical change in the Orientation
	m_Forward_dir = rotated;
	m_Right_dir = cross(m_Forward_dir, m_Up_dir);
}

__device__ float3 Camera::GetRayDir(float2 _uv, float vfovdeg, float width, float height) const
{
	float theta = deg2rad(vfovdeg);
	auto h = tan(theta / 2);
	float viewport_height = 2.0 * h;
	float viewport_width = viewport_height * (width / height);//aspect ratio
	//float viewport_width = viewport_height;

	float3 w = normalize(m_Forward_dir);//front
	float3 u = normalize(cross(w, make_float3(0, 1, 0)));//right
	float3 v = cross(u, w);//up

	float3 horizontal = viewport_width * u;
	float3 vertical = viewport_height * v;

	float3 lower_left_corner = m_Position - horizontal / 2.0f - vertical / 2.0f + w;

	//return float3(lower_left_corner + _uv.x * horizontal + _uv.y * vertical - m_Position);
	return float3((m_Position + w) + _uv.x * horizontal + _uv.y * vertical - m_Position);
}