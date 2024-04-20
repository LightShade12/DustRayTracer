//ONLY INCLUDE ONCE IN RENDERKERNEL
__device__ HitPayload ClosestHit(const Ray& ray, uint32_t obj_idx, float hit_distance, const Sphere* scene_vec) {
	const Sphere* sphere = &(scene_vec[obj_idx]);

	float3 origin = ray.origin - sphere->Position;//apply sphere translation

	HitPayload payload;
	payload.hit_distance = hit_distance;
	payload.world_position = origin + ray.direction * hit_distance;//hit position
	payload.world_normal = normalize(payload.world_position);
	payload.object_idx = obj_idx;

	payload.world_position += sphere->Position;

	return payload;
};
