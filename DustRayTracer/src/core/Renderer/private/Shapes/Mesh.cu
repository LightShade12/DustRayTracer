#include "Mesh.cuh"

#include "core/Common/CudaCommon.cuh"

#include <cuda_runtime.h>

Mesh::Mesh(std::vector<float3> positions, std::vector<float3> normals, uint32_t matidx)
{
	std::vector <Triangle> tris;

	for (size_t i = 0; i < positions.size(); i += 3)
	{
		tris.push_back(Triangle(positions[i], positions[i + 1], positions[i + 2], normals[i / 3], matidx));
	}

	m_trisCount = tris.size();

	cudaMallocManaged((void**)&m_triangles, tris.size() * sizeof(Triangle));
	checkCudaErrors(cudaGetLastError());
	cudaMemcpy(m_triangles, tris.data(), tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
}

Mesh::~Mesh()
{
	cudaFree(m_triangles);
	checkCudaErrors(cudaGetLastError());
}