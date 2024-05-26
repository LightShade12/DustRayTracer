#pragma once
#include "BVHBuilder.cuh"
#include "core/Renderer/private/Kernel/Shaders/Intersection.cuh"
#include "core/Renderer/private/CudaMath/helper_math.cuh"

__device__ HitPayload intersectAABB(const Ray& ray, const Bounds3f& bbox) {
	HitPayload hitInfo;

	float3 invDir = 1.0f / ray.direction;
	float3 t0 = (bbox.pMin - ray.origin) * invDir;
	float3 t1 = (bbox.pMax - ray.origin) * invDir;

	float3 tmin = fminf(t0, t1);
	float3 tmax = fmaxf(t0, t1);

	float tenter = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
	float texit = fminf(fminf(tmax.x, tmax.y), tmax.z);

	if (tenter > texit || texit < 0)
		return hitInfo;

	hitInfo.hit_distance = tenter;
	return hitInfo;
}

//Traversal
__device__ void find_closest_hit(const Ray& ray, const BVHNode* dev_node, HitPayload* closest_hitpayload, bool& debug)
{
	HitPayload workinghitpayload;
	//miss
	//if not intersects
	bool rayInbbox =
		(dev_node->bbox.pMin.x <= ray.origin.x &&
			dev_node->bbox.pMin.y <= ray.origin.y &&
			dev_node->bbox.pMin.z <= ray.origin.z &&
			dev_node->bbox.pMax.x >= ray.origin.x &&
			dev_node->bbox.pMax.y >= ray.origin.y &&
			dev_node->bbox.pMax.z >= ray.origin.z
			);

	if (!rayInbbox)
	{
		workinghitpayload = intersectAABB(ray, dev_node->bbox);
		if (workinghitpayload.hit_distance < 0)
		{
			return;
		}
	}

	//if bbox dist > closest triangle hit dist
	if (closest_hitpayload->primitiveptr != nullptr && workinghitpayload.hit_distance > closest_hitpayload->hit_distance) { return; }

	if (dev_node->leaf)
	{
		//printf("hit dist %.3f\n", workinghitpayload.hit_distance);
		//printf("entered leaf ");
		//TODO: dev_primitive_ptrs_buffer pointer looping
		for (int primIdx = 0; primIdx < dev_node->primitives_count; primIdx++)
		{
			const Triangle* prim = (dev_node->dev_primitive_ptrs_buffer[primIdx]);
			workinghitpayload = Intersection(ray, prim);

			if (workinghitpayload.primitiveptr != nullptr && workinghitpayload.hit_distance < closest_hitpayload->hit_distance)
			{
				//debug = true;
				closest_hitpayload->hit_distance = workinghitpayload.hit_distance;
				closest_hitpayload->primitiveptr = workinghitpayload.primitiveptr;
			}
		}
	}
	else
	{
		//front to back traversal
		if (false)
		{
			HitPayload hit1 = intersectAABB(ray, dev_node->dev_child1->bbox);
			HitPayload hit2 = intersectAABB(ray, dev_node->dev_child2->bbox);

			const BVHNode* first = (hit1.hit_distance <= hit2.hit_distance) ? dev_node->dev_child1 : dev_node->dev_child2;
			const BVHNode* second = (hit1.hit_distance <= hit2.hit_distance) ? dev_node->dev_child2 : dev_node->dev_child1;

			find_closest_hit(ray, first, closest_hitpayload, debug);
			if (closest_hitpayload->hit_distance > hit2.hit_distance)
				find_closest_hit(ray, second, closest_hitpayload, debug);
		}
		else
		{
			find_closest_hit(ray, dev_node->dev_child1, closest_hitpayload, debug);
			find_closest_hit(ray, dev_node->dev_child2, closest_hitpayload, debug);
		}
	}
}