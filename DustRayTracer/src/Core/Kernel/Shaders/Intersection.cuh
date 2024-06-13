#pragma once
#include "Core/Scene/Triangle.cuh"
#include "Core/HitPayload.cuh"
#include "Core/Ray.cuh"
#include "Core/CudaMath/helper_math.cuh"

__device__ ShortHitPayload Intersection(const Ray& ray, const Triangle* triangle);