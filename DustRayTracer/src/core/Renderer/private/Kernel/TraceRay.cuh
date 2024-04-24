#pragma once
#include "core/Renderer/private/Kernel/Ray.cuh"
#include "core/Renderer/private/Kernel/HitPayload.cuh"
#include "core/Renderer/private/Shapes/Mesh.cuh"

__device__ HitPayload TraceRay(const Ray& ray,
	const Mesh* MeshBufferPtr, size_t MeshBufferSize);

__device__ bool RayTest(const Ray& ray,
	const Mesh* MeshBufferPtr, size_t MeshBufferSize);