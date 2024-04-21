#pragma once
#include "core/Renderer/private/Kernel/Ray.cuh"
#include "core/Renderer/private/Kernel/HitPayload.cuh"
#include "core/Renderer/private/Shapes/Triangle.cuh"

__device__ HitPayload TraceRay(const Ray& ray, const Triangle* scene_vector, size_t scene_vector_size);