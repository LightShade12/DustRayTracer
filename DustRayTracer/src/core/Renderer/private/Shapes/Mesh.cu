#include "Mesh.cuh"

#include "core/Common/CudaCommon.cuh"

#include <cuda_runtime.h>

__host__ Mesh::Mesh(std::vector<float3> positions, std::vector<float3> normals, uint32_t matidx)
{
	std::vector <Triangle> tris;

	for (size_t i = 0; i < positions.size(); i += 3)
	{
		tris.push_back(Triangle(Vertex(positions[i]), Vertex(positions[i + 1]), Vertex(positions[i + 2]), normals[i / 3], matidx));
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