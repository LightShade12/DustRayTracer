#pragma once

#include "CudaMath/helper_math.cuh"

struct BSDFSample {
	enum Scatter {
		Absorbed = 0,
		Emitted = 1,
		Reflected = 2,
		Transmitted = 4,
		Diffuse = 8,
		Glossy = 16,
		Specular = 32
	};
	__device__ BSDFSample() = default;
	__device__ BSDFSample(int scatter, float3 f, float3 wi, float pdf) :scatter(scatter), f(f), wi(wi), pdf(pdf) {};

	int scatter;      // Scatter flags
	float3 f;         // BSDF value
	float3 wi;        // Sampled incident light direction
	float pdf;        // Sampled direction PDF

	[[nodiscard]] constexpr bool is(int flag) const noexcept {
		return scatter & flag;
	}
};

class PrincipledBSDF {
	__device__ PrincipledBSDF(float3 albedo = make_float3(1, 0, 1),
		float metallic = 0,
		float roughness = 0.5,
		float transmission = 0,
		float ior = 1.5);

	__device__ float3 f(const float3& wo, const float3& wi);

	__device__ BSDFSample sample_f(const float3& wo, uint32_t& seed);

	__device__ float3 fOpaqueDielectric(const float3& wo, const float3& wi, const float3& albedo);

	__device__ BSDFSample sampleOpaqueDielectric(const float3& wo, const float3& albedo, uint32_t& seed);

	__device__ float pdfOpaqueDielectric(const float3& wo, const float3& wi);

private:
	float3 m_albedo;
	float m_metallicity = 0;
	float m_roughness = 0.5;
	float m_transmission = 0;
	float m_ior = 1.5;
};