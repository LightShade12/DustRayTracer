#include "Intersection.cuh"

__device__ HitPayload Intersection(const Ray& ray, const Triangle* triangle)
{
	HitPayload payload;

	const float EPSILON = 0.000001;
	float3 edge1, edge2, h, s, q;
	float a, f, u, v, t;

	edge1 = triangle->vertex1.position - triangle->vertex0.position;
	edge2 = triangle->vertex2.position - triangle->vertex0.position;

	h = cross(ray.getDirection(), edge2);
	a = dot(edge1, h);
	if (a > -EPSILON && a < EPSILON)
		return payload; // This ray is parallel to this triangle.

	f = 1.0 / a;
	s = ray.getOrigin() - triangle->vertex0.position;
	u = f * dot(s, h);
	if (u < 0.0 || u > 1.0)
		return payload;

	q = cross(s, edge1);
	v = f * dot(ray.getDirection(), q);
	if (v < 0.0 || u + v > 1.0)
		return payload;

	t = f * dot(edge2, q);
	if (t > EPSILON) { // ray intersection
		payload.hit_distance = t;
		payload.primitiveptr = triangle;
		return payload;
	}

	return payload;
};