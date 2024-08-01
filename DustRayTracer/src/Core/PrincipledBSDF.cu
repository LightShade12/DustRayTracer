#include "PrincipledBSDF.cuh"
#include "CudaMath/physical_units.hpp"
#include "CudaMath/Random.cuh"

__device__ float3 sampleCosineHemisphere(float2 u) {
	const float phi = u.x * 2.0f * PI;
	const float sqrtr2 = sqrt(u.y);
	const float x = cos(phi) * sqrtr2;
	const float y = sin(phi) * sqrtr2;
	const float z = sqrt(1.0f - u.y);

	return make_float3(x, y, z);
}

__device__ PrincipledBSDF::PrincipledBSDF(
	float3 albedo,
	float metallic,
	float roughness,
	float transmission,
	float ior) :
	m_albedo(albedo),
	m_metallicity(metallic),
	m_roughness(roughness),
	m_transmission(transmission),
	m_ior(ior)
{}
__device__ float3 PrincipledBSDF::f(const float3& wo, const float3& wi)
{
	float3 val;
	val += fOpaqueDielectric(wo, wi, m_albedo);

	return val;
}
__device__ BSDFSample PrincipledBSDF::sample_f(const float3& wo, uint32_t& seed)
{
	BSDFSample sample;

	sample = sampleOpaqueDielectric(wo, m_albedo, seed);

	return sample;
}

__device__ float3 PrincipledBSDF::fOpaqueDielectric(const float3& wo, const float3& wi, const float3& albedo)
{
	return albedo / PI;
}

__device__ BSDFSample PrincipledBSDF::sampleOpaqueDielectric(const float3& wo, const float3& albedo, uint32_t& seed)
{
	float2 u = make_float2(randomFloat(seed), randomFloat(seed));
	float3 wi = sampleCosineHemisphere(u);
	float cDiffuse = 1 / PI;
	return BSDFSample(BSDFSample::Diffuse | BSDFSample::Reflected,
		fOpaqueDielectric(wo, wi, m_albedo),
		wi,
		pdfOpaqueDielectric(wo, wi));
}

__device__ float PrincipledBSDF::pdfOpaqueDielectric(const float3& wo, const float3& wi)
{
	return abs(wi.y) * 1 / PI;
}