#include "BVH.cuh"


__device__ bool Node::IntersectAABB(const Ray& ray, float t_min, float t_max) const {
	// Implementing AABB-ray intersection algorithm
	float t_near = -FLT_MAX;
	float t_far = FLT_MAX;

	float inv_direction_x = 1.0f / ray.direction.x;
	float inv_direction_y = 1.0f / ray.direction.y;
	float inv_direction_z = 1.0f / ray.direction.z;

	// For x component
	float t1 = (bounds.pMin.x - ray.origin.x) * inv_direction_x;
	float t2 = (bounds.pMax.x - ray.origin.x) * inv_direction_x;
	t_near = fmaxf(t_near, fminf(t1, t2));
	t_far = fminf(t_far, fmaxf(t1, t2));

	// For y component
	t1 = (bounds.pMin.y - ray.origin.y) * inv_direction_y;
	t2 = (bounds.pMax.y - ray.origin.y) * inv_direction_y;
	t_near = fmaxf(t_near, fminf(t1, t2));
	t_far = fminf(t_far, fmaxf(t1, t2));

	// For z component
	t1 = (bounds.pMin.z - ray.origin.z) * inv_direction_z;
	t2 = (bounds.pMax.z - ray.origin.z) * inv_direction_z;
	t_near = fmaxf(t_near, fminf(t1, t2));
	t_far = fminf(t_far, fmaxf(t1, t2));

	return t_far >= t_near && t_far >= t_min && t_near <= t_max;
}