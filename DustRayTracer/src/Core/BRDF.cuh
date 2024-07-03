#pragma once
#include <vector_types.h>
#include <cstdint>

class Material;
class SceneData;

class BRDF {
public:

	__device__ BRDF(const SceneData* scene_data) :m_scene_data(scene_data) {};

	__device__ void setMaterial(const Material* material) {
		m_material = material;
	};

	__device__ float3 evaluateContribution(const float3& wi, const float3& wo,
		const float3& normal, float2 texcoords);

	__device__ float3 importanceSample(const float3& wo, const float3& normal, uint32_t& seed, float& pdf);

private:
	const Material* m_material = nullptr;
	const SceneData* m_scene_data = nullptr;
};