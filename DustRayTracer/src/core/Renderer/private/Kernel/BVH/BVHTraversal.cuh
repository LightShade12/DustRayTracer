#pragma once
#include "BVHBuilder.cuh"
#include "core/Renderer/private/Kernel/Shaders/Intersection.cuh"

__device__ HitPayload intersectAABB(const Ray& ray, const Bounds3f& bbox) {
	HitPayload hitInfo;

	float3 invDir = 1.0f / ray.direction;
	float3 t0 = (bbox.pMin - ray.origin) * invDir;
	float3 t1 = (bbox.pMax - ray.origin) * invDir;

	float3 tmin = fminf(t0, t1);
	float3 tmax = fmaxf(t0, t1);

	float tmin_max = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
	float tmax_min = fminf(fminf(tmax.x, tmax.y), tmax.z);

	if (tmin_max > tmax_min || tmax_min < 0)
		return hitInfo;

	hitInfo.hit_distance = tmin_max;
	return hitInfo;
}

//Traversal
__device__ void find_closest_hit(const Ray& ray, BVHNode* node, HitPayload* closest_hitpayload)
{
	HitPayload hit = intersectAABB(ray, node->bbox);

	//miss
	if (hit.primitiveptr == nullptr || hit.hit_distance > closest_hitpayload->hit_distance)
		return;

	if (node->leaf)
	{
		//TODO: dev_primitive_ptrs_buffer pointer looping
		for (int primIdx = 0; primIdx < node->primitives_count; primIdx++)
		{
			const Triangle* prim = (node->dev_primitive_ptrs_buffer[primIdx]);
			hit = Intersection(ray, prim);
			if (hit.primitiveptr != nullptr && hit.hit_distance < closest_hitpayload->hit_distance)
			{
				closest_hitpayload->primitiveptr = hit.primitiveptr;
				closest_hitpayload->hit_distance = hit.hit_distance;
			}
		}
	}
	else
	{
		//fron to back traversal
		HitPayload hit1 = intersectAABB(ray, node->dev_child1->bbox);
		HitPayload hit2 = intersectAABB(ray, node->dev_child2->bbox);

		BVHNode* first = (hit1.hit_distance <= hit2.hit_distance) ? node->dev_child1 : node->dev_child2;
		BVHNode* second = (hit1.hit_distance <= hit2.hit_distance) ? node->dev_child2 : node->dev_child1;

		find_closest_hit(ray, first, closest_hitpayload);
		if (closest_hitpayload->hit_distance > hit2.hit_distance)
			find_closest_hit(ray, second, closest_hitpayload);
	}
}