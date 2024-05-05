#pragma once
#include "core/Renderer/private/Shapes/Triangle.cuh"
#include "core/Renderer/private/Kernel/HitPayload.cuh"
#include "core/Renderer/private/Kernel/Ray.cuh"
#include "core/Renderer/private/CudaMath/helper_math.cuh"

__device__ HitPayload Intersection(const Ray& ray, const Triangle* triangle);