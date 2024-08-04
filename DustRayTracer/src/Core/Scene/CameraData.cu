#include "CameraData.cuh"
#include "Core/CudaMath/Random.cuh"

__host__ __device__ float deg2rad(float degree)
{
	float const PI = 3.14159265359f;
	return (degree * (PI / 180.f));
}

namespace DustRayTracer {
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