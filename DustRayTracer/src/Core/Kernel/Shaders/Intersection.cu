#include "Intersection.cuh"
#include "Common/physical_units.hpp"

__device__ ShortHitPayload Intersection(const Ray& ray, const Triangle* triangle)
{
	ShortHitPayload payload;

	float3 edge1, edge2, h, s, q;
	float a, f, u, v, t;

	edge1 = triangle->vertex1.position - triangle->vertex0.position;
	edge2 = triangle->vertex2.position - triangle->vertex0.position;

	h = cross(ray.getDirection(), edge2);
	a = dot(edge1, h);
	if (a > -TRIANGLE_EPSILON && a < TRIANGLE_EPSILON)
		return payload; // This ray is parallel to this triangle.

	f = 1.0f / a;
	s = ray.getOrigin() - triangle->vertex0.position;
	u = f * dot(s, h);
	if (u < 0.0f || u > 1.0f)
		return payload;

	q = cross(s, edge1);
	v = f * dot(ray.getDirection(), q);
	if (v < 0.0f || u + v > 1.0f)
		return payload;

	t = f * dot(edge2, q);
	if (t > TRIANGLE_EPSILON) { // ray intersection
		payload.hit_distance = t;
		payload.primitiveptr = triangle;
		return payload;
	}

	return payload;
};