#include "Intersection.cuh"
#include "Common/physical_units.hpp"

__device__ ShortHitPayload Intersection(const Ray& ray, const Triangle* triangle)
{
	ShortHitPayload payload;

	float3 v0v1 = triangle->vertex1.position - triangle->vertex0.position;
	float3 v0v2 = triangle->vertex2.position - triangle->vertex0.position;

	float3 pvec = cross(ray.getDirection(), v0v2);
	float det = dot(v0v1, pvec);
	if (det > -TRIANGLE_EPSILON && det < TRIANGLE_EPSILON)
		return payload; // This ray is parallel to this triangle.

	float invDet = 1.0f / det;
	float3 tvec = ray.getOrigin() - triangle->vertex0.position;
	float u = invDet * dot(tvec, pvec);
	if (u < 0.0f || u > 1.0f)
		return payload;

	float3 qvec = cross(tvec, v0v1);
	float v = invDet * dot(ray.getDirection(), qvec);
	if (v < 0.0f || u + v > 1.0f)
		return payload;

	float t = invDet * dot(v0v2, qvec);
	if (t > TRIANGLE_EPSILON) { // ray intersection
		payload.hit_distance = t;
		payload.primitiveptr = triangle;
		payload.UVW = { 1.0f - u - v, u, v };
		return payload;
	}

	return payload;
};