#include "Camera.cuh"
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

//TODO: wdym OnUpdate hadles this??
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
	float3 finalvel = { transformedVel.x, transformedVel.y,transformedVel.z };//not vel but movedir

	m_Position += m_movement_speed * finalvel * delta;
}

//TODO: esoteric...
__host__ void Camera::Rotate(float4 delta_degrees)
{
	float sin_x = delta_degrees.x;
	float cos_x = delta_degrees.y;
	float sin_y = delta_degrees.z;
	float cos_y = delta_degrees.w;

	float3 rotated = (m_Forward_dir * cos_x) +
		(cross(m_Up_dir, m_Forward_dir) * sin_x) +
		(m_Up_dir * dot(m_Up_dir, m_Forward_dir)) * (1 - cos_x);
	// Calculates upcoming vertical change in the Orientation
	m_Forward_dir = rotated;

	rotated = (m_Forward_dir * cos_y) +
		(cross(m_Right_dir, m_Forward_dir) * sin_y) +
		(m_Right_dir * dot(m_Right_dir, m_Forward_dir)) * (1 - cos_y);
	// Calculates upcoming vertical change in the Orientation
	m_Forward_dir = rotated;
	m_Right_dir = cross(m_Forward_dir, m_Up_dir);
}

//returns normalized dir
__device__ Ray Camera::getRay(float2 _uv, float width, float height, uint32_t& seed) const
{
	float theta = vfov_rad / 2;
	float fov_factor = tan(theta / 2.0f);

	float aspect_ratio = width / height;
	float world_image_plane_height = 2.0f * fov_factor * focus_dist;
	float world_image_plane_width = world_image_plane_height * aspect_ratio;

	float3 forward_dir = normalize(m_Forward_dir);
	float3 right_dir = normalize(cross(forward_dir, make_float3(0, 1, 0)));
	float3 up_dir = cross(right_dir, forward_dir);

	float3 world_image_plane_horizontal_vector = world_image_plane_width * right_dir;
	float3 world_image_plane_vertical_vector = world_image_plane_height * up_dir;

	float2 offset = { randomFloat(seed) - 0.5f, randomFloat(seed) - 0.5f };
	offset *= 0.0035f; // Adjust scale as needed for anti-aliasing

	float defocus_radius = focus_dist * tan(deg2rad(defocus_angle) / 2.0f);
	float3 defocus_disk_u = defocus_radius * right_dir;
	float3 defocus_disk_v = defocus_radius * up_dir;

	float3 rorig;

	if (defocus_angle <= 0)
	{
		rorig = m_Position;
	}
	else
	{
		float2 p = random_in_unit_disk(seed);
		rorig = m_Position + (p.x * defocus_disk_u) + (p.y * defocus_disk_v);
	}

	float3 ray_direction = normalize((forward_dir * focus_dist) +
		((_uv.x + offset.x) * world_image_plane_horizontal_vector) +
		((_uv.y + offset.y) * world_image_plane_vertical_vector) -
		rorig + m_Position);

	return Ray(rorig, ray_direction);
}

__host__ __device__ float deg2rad(float degree)
{
	float const PI = 3.14159265359f;
	return (degree * (PI / 180.f));
}