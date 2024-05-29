#pragma once
#include "Core/Ray.cuh"
#include "Core/HitPayload.cuh"
#include "Core/Scene/Scene.cuh"

__device__ HitPayload TraceRay(const Ray& ray,
	const SceneData* scenedata);

__device__ bool RayTest(const Ray& ray,
	const SceneData* scenedata);