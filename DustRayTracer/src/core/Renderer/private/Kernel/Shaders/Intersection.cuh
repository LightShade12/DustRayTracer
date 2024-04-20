__device__ HitPayload Intersection(const Ray& ray, const Triangle* triangle)
{
	HitPayload payload;

	const float EPSILON = 0.000001;
	float3 edge1, edge2, h, s, q;
	float a, f, u, v, t;

	edge1 = triangle->vertex1 - triangle->vertex0;

	edge2 = triangle->vertex2 - triangle->vertex0;

	h = cross(ray.direction, edge2);
	a = dot(edge1, h);
	if (a > -EPSILON && a < EPSILON)
		return payload; // This ray is parallel to this triangle.

	f = 1.0 / a;
	s.x = ray.origin.x - triangle->vertex0.x;
	s.y = ray.origin.y - triangle->vertex0.y;
	s.z = ray.origin.z - triangle->vertex0.z;
	u = f * dot(s, h);
	if (u < 0.0 || u > 1.0)
		return payload;

	q = cross(s, edge1);
	v = f * dot(ray.direction, q);
	if (v < 0.0 || u + v > 1.0)
		return payload;

	t = f * dot(edge2, q);
	if (t > EPSILON) { // ray intersection
		payload.hit_distance = t;
		return payload;
	}

	return payload;
};