#include "core/Renderer/private/CudaMath/helper_math.cuh"
#include "core/Renderer/private/Shapes/Triangle.cuh"
#include "core/Renderer/private/Shapes/Mesh.cuh"
#include "core/Renderer/private/Kernel/Ray.cuh"
#include <vector>

class Node
{
public:
	__device__ bool IntersectAABB(const Ray& ray, float t_min = 0.0001f, float t_max = FLT_MAX) const;

	Bounds3f bounds;
	Node* children=nullptr;
	size_t childrenCount = 0;
	Mesh* d_Mesh=nullptr;
	int MeshIndex = -1;
};
