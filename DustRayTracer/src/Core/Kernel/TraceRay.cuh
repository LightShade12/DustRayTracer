#pragma once
#include "Core/Ray.cuh"
#include "Core/HitPayload.cuh"
#include "Core/Scene/SceneData.cuh"

__device__ HitPayload traceRay(const Ray& ray,
	const SceneData* scenedata);

__device__ int rayTest(const Ray& ray,
	const SceneData* scenedata);