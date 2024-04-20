
__device__ HitPayload Intersection(const Ray& ray, const Sphere* sphere)
{
	HitPayload payload;

	float3 origin = ray.origin - sphere->Position;

	float a = dot(ray.direction, ray.direction);
	float b = 2.0f * dot(origin, ray.direction);
	float c = dot(origin, origin) - sphere->Radius * sphere->Radius;

	float discriminant = b * b - 4.0f * a * c;
	if (discriminant < 0.0f)
		return payload;//hit dist=-1 

	payload.hit_distance = (-b - sqrt(discriminant)) / (2.0f * a);
	return payload;
};