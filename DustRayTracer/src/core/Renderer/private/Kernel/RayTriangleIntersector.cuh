#include "core/Renderer/private/Kernel/Triangle.cuh"
#include "core/Renderer/private/Kernel/Ray.cuh"
#include "core/Renderer/private/Kernel/HitPayload.cuh"

#include "core/Renderer/private/CudaMath/helper_math.cuh"

__device__ bool rayTriangleIntersect(const Ray& ray, const Triangle& triangle, float& t) {
	const float EPSILON = 0.0000001;
	float3 edge1, edge2, h, s, q;
	float a, f, u, v;

	edge1.x = triangle.vertex1.x - triangle.vertex0.x;
	edge1.y = triangle.vertex1.y - triangle.vertex0.y;
	edge1.z = triangle.vertex1.z - triangle.vertex0.z;
	edge2.x = triangle.vertex2.x - triangle.vertex0.x;
	edge2.y = triangle.vertex2.y - triangle.vertex0.y;
	edge2.z = triangle.vertex2.z - triangle.vertex0.z;

	h = cross(ray.direction, edge2);
	a = dot(edge1, h);

	if (a > -EPSILON && a < EPSILON)
		return false; // This ray is parallel to this triangle.

	f = 1.0 / a;
	s.x = ray.origin.x - triangle.vertex0.x;
	s.y = ray.origin.y - triangle.vertex0.y;
	s.z = ray.origin.z - triangle.vertex0.z;
	u = f * dot(s, h);

	if (u < 0.0 || u > 1.0)
		return false;

	q = cross(s, edge1);
	v = f * dot(ray.direction, q);

	if (v < 0.0 || u + v > 1.0)
		return false;

	// At this stage we can compute t to find out where the intersection point is on the line.
	t = f * dot(edge2, q);
	if (t > EPSILON) // ray intersection
		return true;

	return false; // This means that there is a line intersection but not a ray intersection.
}

bool rayTriangleIntersect(const Ray& ray, const Triangle& triangle, HitPayload& payload) {
    const float EPSILON = 0.000001;
    float3 edge1, edge2, h, s, q;
    float a, f, u, v, t;

    edge1.x = triangle.vertex1.x - triangle.vertex0.x;
    edge1.y = triangle.vertex1.y - triangle.vertex0.y;
    edge1.z = triangle.vertex1.z - triangle.vertex0.z;

    edge2.x = triangle.vertex2.x - triangle.vertex0.x;
    edge2.y = triangle.vertex2.y - triangle.vertex0.y;
    edge2.z = triangle.vertex2.z - triangle.vertex0.z;

    h = cross(ray.direction, edge2);
    a = dot(edge1, h);
    if (a > -EPSILON && a < EPSILON)
        return false; // This ray is parallel to this triangle.

    f = 1.0 / a;
    s.x = ray.origin.x - triangle.vertex0.x;
    s.y = ray.origin.y - triangle.vertex0.y;
    s.z = ray.origin.z - triangle.vertex0.z;
    u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
        return false;

    q = cross(s, edge1);
    v = f * dot(ray.direction, q);
    if (v < 0.0 || u + v > 1.0)
        return false;

    t = f * dot(edge2, q);
    if (t > EPSILON) { // ray intersection
        payload.hit_distance = t;
        payload.world_position = float3{
            ray.origin.x + ray.direction.x * t,
            ray.origin.y + ray.direction.y * t,
            ray.origin.z + ray.direction.z * t
        };
        payload.world_normal = cross(edge1, edge2);
        return true;
    }

    return false;
}