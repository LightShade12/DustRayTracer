#pragma once
#include "Core/Ray.cuh"
#include "Core/HitPayload.cuh"
#include "Core/Scene/Scene.cuh"

__device__ HitPayload traceRay(const Ray& ray,
	const SceneData* scenedata);

__device__ const Triangle* rayTest(const Ray& ray,
	const SceneData* scenedata);