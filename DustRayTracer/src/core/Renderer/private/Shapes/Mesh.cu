#include "Mesh.cuh"

#include "core/Common/CudaCommon.cuh"
#include "core/Renderer/private/CudaMath/helper_math.cuh"

#include <cuda_runtime.h>

__host__ Mesh::Mesh(const std::vector<float3> &positions, const std::vector<float3> &vertex_normals, uint32_t matidx)
{
	std::vector <Triangle> tris;

	//Positions.size() and vertex_normals.size() must be equal!
	for (size_t i = 0; i < positions.size(); i += 3)
	{
		float3 p0 = positions[i + 1] - positions[i];
		float3 p1 = positions[i + 2] - positions[i];
		float3 faceNormal = cross(p0, p1);

		float3 avgVertexNormal = (vertex_normals[i] + vertex_normals[i + 1] + vertex_normals[i + 2]) / 3;
		float ndot = dot(faceNormal, avgVertexNormal);

		float3 surface_normal = (ndot < 0.0f) ? -faceNormal : faceNormal;

		tris.push_back(Triangle(Vertex(positions[i], vertex_normals[i]), Vertex(positions[i + 1], vertex_normals[i + 1]), Vertex(positions[i + 2], vertex_normals[i + 2]),
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