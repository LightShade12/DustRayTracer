#include "core/Renderer/private/CudaMath/helper_math.cuh"
#include "core/Renderer/private/Shapes/Triangle.cuh"
#include "core/Renderer/private/Shapes/Mesh.cuh"
#include "core/Renderer/private/Kernel/Ray.cuh"
#include <vector>

class Node
{
public:
	Node();
	~Node();
	
	__device__ bool Intersect(Ray ray);

	Bounds3f bounds;
	Node* children;
	size_t childrenCount = 0;
	Mesh* d_Mesh;
};

Node::Node()
{
}

Node::~Node()
{
}