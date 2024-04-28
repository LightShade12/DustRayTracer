#pragma once
#include "core/Renderer/private/Kernel/Ray.cuh"
#include "core/Renderer/private/Kernel/HitPayload.cuh"
#include "core/Renderer/private/Shapes/Scene.cuh"

__device__ HitPayload TraceRay(const Ray& ray,
	const SceneData* scenedata);

__device__ bool RayTest(const Ray& ray,
	const SceneData* scenedata);