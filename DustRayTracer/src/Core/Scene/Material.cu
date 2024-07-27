#include "Material.cuh"
#include <cuda_runtime.h>

namespace DustRayTracer {
	HostMaterial::HostMaterial(MaterialData* device_material_data)
	{
		if (device_material_data) {
			m_DeviceMaterialData = device_material_data;
			cudaMemcpy(&m_HostMaterialData, m_DeviceMaterialData, sizeof(MaterialData), cudaMemcpyDeviceToHost);
		}
	}

	void HostMaterial::updateDevice() {
		if (m_DeviceMaterialData) {
			cudaMemcpy(m_DeviceMaterialData, &m_HostMaterialData, sizeof(MaterialData), cudaMemcpyHostToDevice);
		}
	}
}