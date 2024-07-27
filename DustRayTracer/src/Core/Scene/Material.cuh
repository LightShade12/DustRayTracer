#pragma once
#include <vector_types.h>
#include <string>

//Add POM, porosity, sss, alpha
namespace DustRayTracer {
	struct MaterialData
	{
	public:
		MaterialData() = default;

	public:
		char Name[32] = "Unnamed";
		float3 Albedo = { 1,1,1 };
		float3 EmissiveColor = { 0,0,0 };
		float EmissiveScale = 10;
		int AlbedoTextureIndex = -1;
		int RoughnessTextureIndex = -1;
		int NormalTextureIndex = -1;
		int EmissionTextureIndex = -1;
		float Reflectance = 0.5;// default: F0=0.16
		float Metallicity = 0;
		float Roughness = 0.8f;
		float NormalMapScale = 1.0f;
	};
	class HostMaterial {
	public:
		HostMaterial() = default;
		HostMaterial(MaterialData* device_material_data);

		void updateDevice();
		MaterialData* getDeviceMaterialData() { return m_DeviceMaterialData; };
		MaterialData getHostMaterialData() const { return m_HostMaterialData; };
		const char* getName() { return m_HostMaterialData.Name; };
		char* getNamePtr() { return (m_HostMaterialData.Name); };

		void setAlbedo(float3 albedo) { m_HostMaterialData.Albedo = albedo; };
		float3 getAlbedo() const { return m_HostMaterialData.Albedo; };
		float* getAlbedoPtr() { return &(m_HostMaterialData.Albedo.x); };

		void setEmissiveColor(float3 emissiveColor) { m_HostMaterialData.EmissiveColor = emissiveColor; };
		float3 getEmissiveColor() const { return m_HostMaterialData.EmissiveColor; };
		float* getEmissiveColorPtr() { return &(m_HostMaterialData.EmissiveColor.x); };

		void setEmissiveScale(float emissiveScale) { m_HostMaterialData.EmissiveScale = emissiveScale; };
		float getEmissiveScale() const { return m_HostMaterialData.EmissiveScale; };
		float* getEmissiveScalePtr() { return &(m_HostMaterialData.EmissiveScale); };

		void setReflectance(float reflectance) { m_HostMaterialData.Reflectance = reflectance; };
		float getReflectance() const { return m_HostMaterialData.Reflectance; };
		float* getReflectancePtr() { return &(m_HostMaterialData.Reflectance); };

		void setMetallicity(float metallicity) { m_HostMaterialData.Metallicity = metallicity; };
		float getMetallicity() const { return m_HostMaterialData.Metallicity; };
		float* getMetallicityPtr() { return &(m_HostMaterialData.Metallicity); };

		void setRoughness(float roughness) { m_HostMaterialData.Roughness = roughness; };
		float getRoughness() const { return m_HostMaterialData.Roughness; };
		float* getRoughnessPtr() { return &(m_HostMaterialData.Roughness); };

		void setNormalMapScale(float normalMapScale) { m_HostMaterialData.NormalMapScale = normalMapScale; };
		float getNormalMapScale() const { return m_HostMaterialData.NormalMapScale; };
		float* getNormalMapScalePtr() { return &(m_HostMaterialData.NormalMapScale); };

		void setAlbedoTextureIndex(int index) { m_HostMaterialData.AlbedoTextureIndex = index; };
		int getAlbedoTextureIndex() const { return m_HostMaterialData.AlbedoTextureIndex; };

		void setRoughnessTextureIndex(int index) { m_HostMaterialData.RoughnessTextureIndex = index; };
		int getRoughnessTextureIndex() const { return m_HostMaterialData.RoughnessTextureIndex; };

		void setNormalTextureIndex(int index) { m_HostMaterialData.NormalTextureIndex = index; };
		int getNormalTextureIndex() const { return m_HostMaterialData.NormalTextureIndex; };

		void setEmissionTextureIndex(int index) { m_HostMaterialData.EmissionTextureIndex = index; };
		int getEmissionTextureIndex() const { return m_HostMaterialData.EmissionTextureIndex; };
	private:
		MaterialData* m_DeviceMaterialData = nullptr;
		MaterialData m_HostMaterialData;
	};
}