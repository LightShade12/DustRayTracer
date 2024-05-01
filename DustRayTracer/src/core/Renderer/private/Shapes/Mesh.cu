#include "Mesh.cuh"

#include "core/Editor/Common/CudaCommon.cuh"
#include "core/Renderer/private/CudaMath/helper_math.cuh"

#include <cuda_runtime.h>

Mesh::Mesh(const std::vector<float3>& positions,
	const std::vector<float3>& vertex_normals, const std::vector<float2>& vertex_UVs, uint32_t matidx)
{
	std::vector <Triangle> tris;

	//Positions.size() and vertex_normals.size() must be equal!
	for (size_t i = 0; i < positions.size(); i += 3)
	{
		//surface normal construction
		float3 p0 = positions[i + 1] - positions[i];
		float3 p1 = positions[i + 2] - positions[i];
		float3 faceNormal = cross(p0, p1);

		float3 avgVertexNormal = (vertex_normals[i] + vertex_normals[i + 1] + vertex_normals[i + 2]) / 3;
		float ndot = dot(faceNormal, avgVertexNormal);

		float3 surface_normal = (ndot < 0.0f) ? -faceNormal : faceNormal;

		//bounding box
		for (size_t j = i; j < i + 3; j++)
		{
			if (Bounds.pMax.x < positions[j].x)Bounds.pMax.x = positions[j].x;
			if (Bounds.pMax.y < positions[j].y)Bounds.pMax.y = positions[j].y;
			if (Bounds.pMax.z < positions[j].z)Bounds.pMax.z = positions[j].z;

			if (Bounds.pMin.x > positions[j].x)Bounds.pMin.x = positions[j].x;
			if (Bounds.pMin.y > positions[j].y)Bounds.pMin.y = positions[j].y;
			if (Bounds.pMin.z > positions[j].z)Bounds.pMin.z = positions[j].z;
		}

		tris.push_back(Triangle(
			Vertex(positions[i], vertex_normals[i], vertex_UVs[i]),
			Vertex(positions[i + 1], vertex_normals[i + 1], vertex_UVs[i + 1]),
			Vertex(positions[i + 2], vertex_normals[i + 2], vertex_UVs[i + 2]),
			normalize(surface_normal), matidx));
	}

	m_trisCount = tris.size();

	cudaMallocManaged((void**)&m_dev_triangles, tris.size() * sizeof(Triangle));
	checkCudaErrors(cudaGetLastError());
	cudaMemcpy(m_dev_triangles, tris.data(), tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
}

__host__ void Mesh::Cleanup()
{
	cudaFree(m_dev_triangles);
	checkCudaErrors(cudaGetLastError());
}