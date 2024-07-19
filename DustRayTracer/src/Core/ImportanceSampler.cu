#include "ImportanceSampler.cuh"

#include "BRDF.cuh"
#include "Scene/SceneData.cuh"
#include "Scene/Texture.cuh"
#include "Scene/Material.cuh"
#include "Common/physical_units.hpp"

#include "Core/CudaMath/Random.cuh"
#include "Core/CudaMath/helper_math.cuh"

__device__ float3 sampleCosineWeightedHemisphere(float3 normal, float2 xi) {
	// Generate a cosine-weighted direction in the local frame
	float phi = 2.0f * PI * xi.x;
	float cosTheta = sqrtf(xi.y);//TODO: might have to switch with sinTheta
	float sinTheta = sqrtf(1.0f - xi.y);

	float3 H;
	H.x = sinTheta * cosf(phi);
	H.y = sinTheta * sinf(phi);
	H.z = cosTheta;

	// Create an orthonormal basis (tangent, bitangent, normal)
	float3 up = fabs(normal.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
	float3 tangent = normalize(cross(up, normal));
	float3 bitangent = cross(normal, tangent);

	// Transform the sample direction from local space to world space
	return normalize(tangent * H.x + bitangent * H.y + normal * H.z);
}

//clamped roughness
__device__ float getPDF(ImportanceSampleData importancedata, float3 out_dir, float3 normal, const Material& material,
	const SceneData& scene_data, float2 texture_uv) {
	if (importancedata.specular) {
		float roughness = material.Roughness;
		//roughness-metallic texture
		if (material.RoughnessTextureIndex >= 0) {
			float3 col = scene_data.DeviceTextureBufferPtr[material.RoughnessTextureIndex].getPixel(texture_uv, true);
			roughness = col.y * material.Roughness;
		}
		float3 H = normalize(out_dir + importancedata.sampleDir);
		float VoH = fmaxf(dot(out_dir, H), 0.0f);
		float NoH = fmaxf(dot(normal, H), 0.0f);
		float D = D_GGX(NoH, roughness);
		return (VoH > 0.0) ? (D * NoH) / (4.0f * VoH) : 0.0;
	}
	else {
		return fmaxf(dot(normal, importancedata.sampleDir), 0.0f) / PI;
	}
}

__device__ static float luma(float3 color) {
	return dot(color, make_float3(0.2126f, 0.7152f, 0.0722f));
}

__device__ float3 sphericalToCartesian(float theta, float phi) {
	float3 wm;
	wm.x = sin(theta) * cos(phi);
	wm.y = cos(theta);
	wm.z = sin(theta) * sin(phi);
	return wm;
	//return normalize(wm);
}

__device__ void ImportanceSampleGgxD(uint32_t& seed, float3 normal,
	float3 wo, float roughness, float3& halfvec,
	float3& wi, float3& reflectance, float3 f0)
{
	float a = roughness * roughness;
	float a2 = a * a;
	// -- Generate uniform random variables between 0 and 1
	float e0 = randomFloat(seed);
	float e1 = randomFloat(seed);

	// -- Calculate theta and phi for our microfacet normal wm by
	// -- importance sampling the Ggx distribution of normals
	float theta = acosf(sqrtf((1.0f - e0) / ((a2 - 1.0f) * e0 + 1.0f)));
	float phi = 2 * PI * e1;

	// -- Convert from spherical to Cartesian coordinates
	float3 wm = sphericalToCartesian(theta, phi);
	halfvec = wm;

	// -- Calculate wi by reflecting wo about wm
	wi = 2.0f * dot(wo, wm) * wm - wo;//TODO: maybe use reflect

	//REDUNDANT
	// -- Ensure our sample is in the upper hemisphere
	// -- Since we are in tangent space with a y-up coordinate
	// -- system BsdfNDot(wi) simply returns wi.y
	if (dot(normal, wi) > 0.0f && dot(wi, wm) > 0.0f)
	{
		float dotWiWm = dot(wi, wm);

		// -- calculate the reflectance to multiply by the energy
		// -- retrieved in direction wi
		float3 F = fresnelSchlick(dotWiWm, f0);
		//float G = SmithGGXMaskingShadowing(wi, wo, normal, a2);
		float G = G2_Smith(wo, wi, normal, roughness);
		float weight = fabsf(dot(wo, wm))
			/ (dot(normal, wo) * dot(normal, wm));

		reflectance = F * G * weight;
	}
	else {
		reflectance = make_float3(0);
	}
}

__device__ void computeTangentBasis(const float3& normal, float3& tangent, float3& bitangent) {
	// Assume normal is already normalized
	if (fabsf(normal.y) > 0.9999f) {
		// Handle the singularity case
		tangent = make_float3(1.0f, 0.0f, 0.0f);
	}
	else {
		tangent = normalize(cross(make_float3(0.0f, 1.0f, 0.0f), normal));
	}
	bitangent = cross(normal, tangent);
}

__device__ float3 transformWorldToTangent(const float3& vec, const float3& normal, const float3& tangent, const float3& bitangent) {
	return make_float3(dot(vec, tangent), dot(vec, normal), dot(vec, bitangent));
}

__device__ float3 transformTangentToWorld(const float3& vec, const float3& normal, const float3& tangent, const float3& bitangent) {
	return vec.x * tangent + vec.y * normal + vec.z * bitangent;
}

//clamped roughness
__device__ ImportanceSampleData importanceSampleBRDF(float3 normal, float3 viewDir, const Material& material, uint32_t& seed, float& pdf,
	float3& throughput, const SceneData& scene_data, float2 texture_uv)
{
	//TODO: bad weighting to balance both samples
	//TODO: bad energy conservation
	float roughness = material.Roughness;
	float metallicity = material.Metallicity;
	float3 basecolor = material.Albedo;

	if (material.AlbedoTextureIndex >= 0)basecolor = scene_data.DeviceTextureBufferPtr[material.AlbedoTextureIndex].getPixel(texture_uv);
	//roughness-metallic texture
	if (material.RoughnessTextureIndex >= 0) {
		float3 col = scene_data.DeviceTextureBufferPtr[material.RoughnessTextureIndex].getPixel(texture_uv, true);
		roughness = col.y * material.Roughness;
		metallicity = col.z * material.Metallicity;
	}

	float3 f0 = make_float3(0.16f * (material.Reflectance * material.Reflectance));//f0=0.04 for most mats
	f0 = lerp(f0, basecolor, metallicity);

	float NoV = fmaxf(dot(normal, viewDir), 0.0);
	bool is_specular_ray = false;
	float3 sampleDir;
	float3 halfvector;

	//float3 F_refl_blend = fresnelSchlick(VoH, f0);
	//float3 F_trans_blend = 1 - F_refl_blend;//may cause problem with metals

	float random_probability = randomFloat(seed);//fresnel rng
	//float specular_probability = (1 - roughness) * luma(F_refl_blend);// / (luma(F_refl_blend) + luma(F_trans_blend));
	float specular_probability = .5f;
	//specular_probability = 0.f;

	if (specular_probability > random_probability)
	{
		float3 L;
		float3 refl;//this is just preemptive specular brdf eval; not needed
		float3 tang, bitan;
		computeTangentBasis(normal, tang, bitan);

		float3 tangentspaceviewdir = transformWorldToTangent(viewDir, normal, tang, bitan);

		ImportanceSampleGgxD(seed, make_float3(0, 1, 0), tangentspaceviewdir, roughness,
			halfvector, L, refl, f0);

		L = transformTangentToWorld(L, normal, tang, bitan);
		halfvector = transformTangentToWorld(halfvector, normal, tang, bitan);

		float NoL = fmaxf(0.f, dot(normal, L));
		float NoH = fmaxf(dot(normal, halfvector), 0.0);
		float VoH = fmaxf(dot(viewDir, halfvector), 0.0001);

		if (NoL > 0 && NoV > 0)
		{
			// See the Heitz paper referenced above for the estimator explanation.
			//   (BRDF / PDF) = F * G2(V, L) / G1(V)
			// Assume G2 = G1(V) * G1(L) here and simplify that expression to just G1(L).
			// Specular sample using GGX

			//float G1_NoL = G1_GGX_Schlick(NoL, roughness);
			//float3 F = fresnelSchlick(VoH, f0);
			//throughput *= G1_NoL * F;

			float D = D_GGX(NoH, roughness);
			pdf = (VoH > 0.0) ? (D * NoH) / (4.0f * VoH) : 0.0;

			throughput *= (1 / (specular_probability));

			is_specular_ray = true;
			sampleDir = normalize(L);
		}
	}

	if (!is_specular_ray)
	{
		// Diffuse sample using cosine-weighted hemisphere
		float3 sphereDir = sampleCosineWeightedHemisphere(normal, { randomFloat(seed),randomFloat(seed) });
		sampleDir = normalize(sphereDir);
		throughput *= (1.0f / (1.0f - specular_probability));

		halfvector = normalize(sampleDir + viewDir);
		float VoH = fmaxf(0.f, dot(viewDir, halfvector));

		float3 F = fresnelSchlick(VoH, f0);

		throughput *= (1.0f - F);//TODO: maybe energy violation culprit?
		pdf = dot(normal, sampleDir) / PI;
	}
	if (specular_probability > 0 && specular_probability < 1)throughput /= 2;

	return ImportanceSampleData(sampleDir, is_specular_ray);
}