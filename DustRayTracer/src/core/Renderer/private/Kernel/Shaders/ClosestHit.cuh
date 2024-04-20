//ONLY INCLUDE ONCE IN RENDERKERNEL
__device__ HitPayload ClosestHit(const Ray& ray, uint32_t obj_idx, float hit_distance, const Triangle* scene_vec) {
	const Triangle* triangle = &(scene_vec[obj_idx]);

	HitPayload payload;
	float3 edge1, edge2;

	edge1 = triangle->vertex1 - triangle->vertex0;

	edge2 = triangle->vertex2 - triangle->vertex0;

	float3 origin = ray.origin; //- sphere->Position;//apply sphere translation

	payload.hit_distance = hit_distance;
	payload.world_position = origin + ray.direction * hit_distance;//hit position
	payload.object_idx = obj_idx;

	payload.world_normal = -normalize(cross(edge1, edge2));

	//payload.world_position += sphere->Position;

	return payload;
};