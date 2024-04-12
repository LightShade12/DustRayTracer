#include "RenderKernel.hpp"

#include "core/Renderer/private/Shapes/Scene.cuh"
#include "core/Renderer/private/Camera/Camera.hpp"
#include "core/Renderer/private/CudaMath/helper_math.hpp"//check if this requires definition activation

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#define __CUDACC__ // used to get surf2d indirect functions;not how it should be done
#include <surface_indirect_functions.h>

struct Ray {
	float3 origin;
	float3 direction;
};

struct Triangle {
	float3 vertex0, vertex1, vertex2;
};

__device__ bool rayTriangleIntersect(const Ray& ray, const Triangle& triangle, float& t) {
	const float EPSILON = 0.0000001;
	float3 edge1, edge2, h, s, q;
	float a, f, u, v;

	edge1.x = triangle.vertex1.x - triangle.vertex0.x;
	edge1.y = triangle.vertex1.y - triangle.vertex0.y;
	edge1.z = triangle.vertex1.z - triangle.vertex0.z;
	edge2.x = triangle.vertex2.x - triangle.vertex0.x;
	edge2.y = triangle.vertex2.y - triangle.vertex0.y;
	edge2.z = triangle.vertex2.z - triangle.vertex0.z;

	h = cross(ray.direction, edge2);
	a = dot(edge1, h);

	if (a > -EPSILON && a < EPSILON)
		return false; // This ray is parallel to this triangle.

	f = 1.0 / a;
	s.x = ray.origin.x - triangle.vertex0.x;
	s.y = ray.origin.y - triangle.vertex0.y;
	s.z = ray.origin.z - triangle.vertex0.z;
	u = f * dot(s, h);

	if (u < 0.0 || u > 1.0)
		return false;

	q = cross(s, edge1);
	v = f * dot(ray.direction, q);

	if (v < 0.0 || u + v > 1.0)
		return false;

	// At this stage we can compute t to find out where the intersection point is on the line.
	t = f * dot(edge2, q);
	if (t > EPSILON) // ray intersection
		return true;

	return false; // This means that there is a line intersection but not a ray intersection.
}

//Render Kernel
__global__ void kernel(cudaSurfaceObject_t _surfobj, int max_x, int max_y, Camera* cam, const Sphere* sceneVector, size_t sceneVectorSize)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;

	float2 uv = { (float(i) / max_x) ,(float(j) / max_y) };

	//uv.x *= ((float)max_x / (float)max_y);
	//uv.x = uv.x * 2.f - ((float)max_x / (float)max_y);
	//uv.y = uv.y * 2.f - 1.f;
	uv = uv * 2 - 1;

	float3 rayOrigin = (cam)->m_Position;
	float3 rayDirection = cam->GetRayDir(uv, 30, max_x, max_y);
	//float radius = 0.5f;

	uchar4 color = { 0,0,0,255 };

	if (sceneVectorSize == 0)
	{
		float a = 0.5 * (1 + (normalize(rayDirection)).y);

		float3 col1 = { 0.5,0.7,1.0 };
		float3 col2 = { 1,1,1 };
		float3 fcol = (float(1 - a) * col2) + (a * col1);
		color.x = 255 * fcol.x;
		color.y = 255 * fcol.y;
		color.z = 255 * fcol.z;
		surf2Dwrite(color, _surfobj, i * 4, j);
		return;
	}

	const Sphere* closestSphere = nullptr;
	float hitDistance = FLT_MAX;//std::numeric_limits<float>::max();

	for (int i = 0; i < sceneVectorSize; i++)
	{
		const Sphere sphere = sceneVector[i];
		float3 origin = rayOrigin - sphere.Position;

		float a = dot(rayDirection, rayDirection);
		float b = 2.0f * dot(origin, rayDirection);
		float c = dot(origin, origin) - sphere.Radius * sphere.Radius;

		// Quadratic forumula discriminant:
		// b^2 - 4ac

		float discriminant = b * b - 4.0f * a * c;
		if (discriminant < 0.0f)
			continue;

		// Quadratic formula:
		// (-b +- sqrt(discriminant)) / 2a

		// float t0 = (-b + glm::sqrt(discriminant)) / (2.0f * a); // Second hit distance (currently unused)
		float closestT = (-b - sqrt(discriminant)) / (2.0f * a);
		if (closestT < hitDistance)
		{
			hitDistance = closestT;
			closestSphere = &sphere;
		}
	}

	if (closestSphere == nullptr)
	{
		float a = 0.5 * (1 + (normalize(rayDirection)).y);

		float3 col1 = { 0.5,0.7,1.0 };
		float3 col2 = { 1,1,1 };
		float3 fcol = (float(1 - a) * col2) + (a * col1);
		color.x = 255 * fcol.x;
		color.y = 255 * fcol.y;
		color.z = 255 * fcol.z;
		surf2Dwrite(color, _surfobj, i * 4, j);
		return;
	}

	float3 origin = rayOrigin - closestSphere->Position;//apply sphere translation
	float3 hitPoint = origin + rayDirection * hitDistance;
	float3 normal = normalize(hitPoint);
	
	float3 lightDir = normalize(make_float3(-1, -1, -1));
	float lightIntensity = max(dot(normal, -lightDir), 0.0f); // == cos(angle)

	float3 sphereColor = closestSphere->Albedo;
	sphereColor *= lightIntensity;

	color.x = 255 * sphereColor.x;
	color.y = 255 * sphereColor.y;
	color.z = 255 * sphereColor.z;

	surf2Dwrite(color, _surfobj, i * 4, j);
};

void InvokeRenderKernel(
	cudaSurfaceObject_t surfaceobj, uint32_t width, uint32_t height,
	dim3 _blocks, dim3 _threads, Camera* cam, const Scene& scene)
{
	const Sphere* DeviceSceneVector = thrust::raw_pointer_cast(scene.m_Spheres.data());
	kernel << < _blocks, _threads >> > (surfaceobj, width, height, cam, DeviceSceneVector, scene.m_Spheres.size());
}