#include "ImportanceSampler.cuh"

#include "BRDF.cuh"
#include "Scene/Material.cuh"
#include "Common/physical_units.hpp"

#include "Core/CudaMath/Random.cuh"
#include "Core/CudaMath/helper_math.cuh"

__device__ float3 sampleGGX(float3 normal, float roughness, float2 xi) {
	float alpha = roughness * roughness;

	float phi = 2.0f * PI * xi.x;
	float cosTheta = sqrtf((1.0f - xi.y) / (1.0f + (alpha * alpha - 1.0f) * xi.y));
	float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

	float3 H;
	H.x = sinTheta * cosf(phi);
	H.y = sinTheta * sinf(phi);
	H.z = cosTheta;

	float3 up = fabs(normal.z) < 0.999 ? make_float3(0.0, 0.0, 1.0) : make_float3(1.0, 0.0, 0.0);
	float3 tangent = normalize(cross(up, normal));
	float3 bitangent = cross(normal, tangent);

	return normalize(tangent * H.x + bitangent * H.y + normal * H.z);
}
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

__device__ float getPDF(float3 out_dir, float3 in_dir, float3 normal, float roughness, float3 halfvector, bool specular) {
	if (specular) {
		float VoH = fmaxf(dot(out_dir, halfvector), 0.0f);
		float NoH = fmaxf(dot(normal, halfvector), 0.0f);
		float D = D_GGX(NoH, roughness);
		return  D * NoH / (4.0f * VoH);
	}
	else {
		return fmaxf(dot(normal, in_dir), 0.0f) / PI;
	}
}

__device__ static float luma(float3 color) {
	return dot(color, make_float3(0.2126f, 0.7152f, 0.0722f));
}

__device__ float ImportanceSampleGGX_VNDF_PDF(float roughness, float3 N, float3 V, float3 L)
{
	float clamped_roughness = fmaxf(roughness, 0.01);
	float3 H = normalize(L + V);
	float NoH = clamp(dot(N, H), 0.f, 1.f);
	float VoH = clamp(dot(V, H), 0.f, 1.f);
	float alpha = clamped_roughness * clamped_roughness;
	float alpha2 = alpha * alpha;
	float NoH2 = NoH * NoH;

	float b = (NoH2 * (alpha2 - 1.0) + 1.0);
	//float b = NoH2 * alpha2 + (1 - NoH2);

	float D = alpha2 / (PI * (b * b));

	return (VoH > 0.0) ? D / (4.0 * VoH) : 0.0;
}

__device__ float3 sphericalToCartesian(float theta, float phi) {
	float3 wm;
	wm.x = sin(theta) * cos(phi);
	wm.y = cos(theta);
	wm.z = sin(theta) * sin(phi);
	return wm;
	//return normalize(wm);
}

__device__ float ggxPdf(float alpha, float theta_m) {
	//alt
	//float D = D_GGX(NoH, roughness);
	//return (D * NoH) / (4.0f * VoH);

	// Calculate cos(theta_m) and sin(theta_m)
	float cosThetaM = cosf(theta_m);
	float sinThetaM = sinf(theta_m);

	// Compute the numerator and the denominator separately
	float alpha2 = alpha * alpha;
	float cosThetaM2 = cosThetaM * cosThetaM;
	float denom = PI * powf((alpha2 - 1) * cosThetaM2 + 1, 2);

	// Compute the PDF value
	float pdf = (alpha2 * cosThetaM * sinThetaM) / denom;

	return pdf;
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
		//reflectance = make_float3(1);
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

__device__ ImportanceSampleData importanceSampleBRDF(float3 normal, float3 viewDir, const Material& material, uint32_t& seed, float& pdf, float3& throughput)
{
	float roughness = clamp(material.Roughness, 0.001f, 1.f);
	float metallicity = material.Metallicity;
	float3 f0 = make_float3(0.16f * (material.Reflectance * material.Reflectance));//f0=0.04 for most mats
	f0 = lerp(f0, material.Albedo, metallicity);

	float NoV = fmaxf(dot(normal, viewDir), 0.0);
	bool is_specular_ray = false;
	float3 sampleDir;
	float3 halfvector;

	//float3 F_refl_blend = fresnelSchlick(VoH, f0);
	//float3 F_trans_blend = 1 - F_refl_blend;//may cause problem with metals

	float random_probability = randomFloat(seed);//fresnel rng
	//float specular_probability = (1 - roughness) * luma(F_refl_blend);// / (luma(F_refl_blend) + luma(F_trans_blend));
	float specular_probability = 0.5f;
	//specular_probability = 0;

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
		float VoH = fmaxf(dot(viewDir, halfvector), 0.0);

		if (NoL > 0 && NoV > 0)
		{
			// See the Heitz paper referenced above for the estimator explanation.
			//   (BRDF / PDF) = F * G2(V, L) / G1(V)
			// Assume G2 = G1(V) * G1(L) here and simplify that expression to just G1(L).
			// Specular sample using GGX

			//float G1_NoL = G1_GGX_Schlick(NoL, roughness);
			//float3 F = fresnelSchlick(VoH, f0);
			//throughput *= G1_NoL * F;
			//throughput *= refl;

			float D = D_GGX(NoH, roughness);
			pdf = (D * NoH) / (4.0f * VoH);

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

		throughput *= (1.0f - F);
		pdf = dot(normal, sampleDir) / PI;
		//pdf = 1;
	}
	if (specular_probability > 0 && specular_probability < 1)throughput /= 2;

	return ImportanceSampleData(sampleDir, halfvector, is_specular_ray);
}
//

//__device__ float3 ImportanceSampleGGX_VNDF(float2 u, float roughness, float3 V, Matrix3x3_d basis)
//{
//	float alpha = roughness * roughness;
//	float pt_ndf_trim = 1;
//	float3 Ve = -make_float3(dot(V, make_float3(basis.m_row[0])), dot(V, make_float3(basis.m_row[2])), dot(V, make_float3(basis.m_row[1])));
//
//	float3 Vh = normalize(make_float3(alpha * Ve.x, alpha * Ve.y, Ve.z));
//
//	float lensq = (Vh.x * Vh.x) + (Vh.y * Vh.y);
//	float3 T1 = lensq > 0.0 ? make_float3(-Vh.y, Vh.x, 0.0) * (1 / sqrt(lensq)) : make_float3(1.0, 0.0, 0.0);
//	float3 T2 = cross(Vh, T1);
//
//	float r = sqrt(u.x * pt_ndf_trim);
//	float phi = 2.0 * PI * u.y;
//	float t1 = r * cos(phi);
//	float t2 = r * sin(phi);
//	float s = 0.5 * (1.0 + Vh.z);
//	t2 = (1.0 - s) * sqrt(1.0 - (t1 * t1)) + s * t2;
//
//	float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - (t1 * t1) - (t2 * t2))) * Vh;
//
//	// Tangent space H
//	float3 Ne = make_float3(alpha * Nh.x, max(0.0, Nh.z), alpha * Nh.y);
//
//	// World space H
//	return normalize(basis * Ne);
//}

//__device__ Matrix3x3_d construct_ONB_frisvad(float3 normal)
//{
//	Matrix3x3_d ret;
//	ret.m_row[1] = make_float4(normal);
//	if (normal.z < -0.999805696f) {
//		ret.m_row[0] = make_float4(0.0f, -1.0f, 0.0f, 1);
//		ret.m_row[2] = make_float4(-1.0f, 0.0f, 0.0f, 1);
//	}
//	else {
//		float a = 1.0f / (1.0f + normal.z);
//		float b = -normal.x * normal.y * a;
//		ret.m_row[0] = make_float4(1.0f - normal.x * normal.x * a, b, -normal.x, 1);
//		ret.m_row[2] = make_float4(b, 1.0f - normal.y * normal.y * a, -normal.y, 1);
//	}
//	return ret;
//}

//__device__ ImportanceSampleData importanceSample(uint32_t& seed, float3 w_o, float3 normal, float roughness, float metallic, float3 f0, float3& throughput)
//{
//	bool is_specular_ray = false;
//	float3 sampledir{};
//	float NoV = fmaxf(0.f, dot(normal, w_o));
//
//	{
//		float2 rng3 = make_float2(randomFloat(seed), randomFloat(seed));
//
//		float rng_frensel = randomFloat(seed);
//
//		float specular_pdf = 0;
//
//		specular_pdf = (metallic == 1) ? 1.0 : 0.5;
//
//		if (rng_frensel < specular_pdf)
//		{
//			Matrix3x3_d basis = construct_ONB_frisvad(normal);
//
//			// Sampling of normal distribution function to compute the reflected ray.
//			// See the paper "Sampling the GGX Distribution of Visible Normals" by E. Heitz,
//			// Journal of Computer Graphics Techniques Vol. 7, No. 4, 2018.
//			// http://jcgt.org/published/0007/04/01/paper.pdf
//
//			float3 N = normal;
//			float3 V = w_o;
//			float3 H = ImportanceSampleGGX_VNDF(rng3, roughness, V, basis);
//			float3 L = reflect(V, H);
//
//			float NoL = fmaxf(0.f, dot(N, L));
//			float NoH = fmaxf(0.f, dot(N, H));
//			float VoH = fmaxf(0.f, dot(V, H));
//
//			if (NoL > 0 && NoV > 0)
//			{
//				// See the Heitz paper referenced above for the estimator explanation.
//				//   (BRDF / PDF) = F * G2(V, L) / G1(V)
//				// Assume G2 = G1(V) * G1(L) here and simplify that expression to just G1(L).
//
//				float G1_NoL = G1_GGX_Schlick(NoL, roughness);
//				float3 F = fresnelSchlick(VoH, f0);
//
//				throughput *= G1_NoL * F;
//
//				throughput *= 1 / specular_pdf;
//				is_specular_ray = true;
//				sampledir = normalize(L);
//			}
//			return ImportanceSampleData(sampledir, H, is_specular_ray);
//		}
//
//		if (!is_specular_ray)
//		{
//			float3 basis_normal, dir_sphere;
//
//			{
//				dir_sphere = sampleCosineWeightedHemisphere(normal, { randomFloat(seed),randomFloat(seed) });
//				basis_normal = normal;
//			}
//
//			Matrix3x3_d basis = construct_ONB_frisvad(basis_normal);
//			sampledir = normalize(basis * dir_sphere);
//			throughput *= 1 / (1 - specular_pdf);
//
//			float3 L = sampledir;
//			float3 V = w_o;
//			float3 H = normalize(V + L);
//			float VoH = fmaxf(0.f, dot(V, H));
//
//			float3 F = fresnelSchlick(VoH, f0);
//
//			throughput *= 1.0f - F;
//			return ImportanceSampleData(sampledir, H, is_specular_ray);
//		}
//	}
//}

//__device__ float3 importanceSampleBRDF(float3 normal, float3 viewDir, const Material& material, uint32_t& seed, float& pdf, const SceneData& scene_data, float2 texture_uv) {
//	float roughness = material.Roughness;
//	float metallicity = material.Metallicity;
//	float3 sampleDir;
//	float2 xi = make_float2(randomFloat(seed), randomFloat(seed)); // uniform rng sample
//
//	float3 specular_H = sampleGGX(normal, roughness, xi);
//	float VoH = fmaxf(dot(viewDir, specular_H), 0.0);
//	float NoH = fmaxf(dot(normal, specular_H), 0.0);
//
//	float3 F_refl_blend = fresnelSchlick(VoH,
//		lerp(make_float3(0.16 * (material.Reflectance * material.Reflectance)), material.Albedo, metallicity));
//	float3 F_trans_blend = 1 - F_refl_blend;//may cause problem with metals
//
//	float random_probability = randomFloat(seed);
//	float specular_probability = luma(F_refl_blend) / luma(F_refl_blend + F_trans_blend);//luma
//	//float t = 0.25; // specular probability threshold
//
//	if (specular_probability > random_probability) {
//		// Specular sample using GGX
//		//xi.x /= t;
//		//float3 H = sampleGGX(normal, roughness, xi);
//		sampleDir = reflect(-1.f * viewDir, specular_H);
//
//		float D = D_GGX(NoH, roughness);
//		pdf = D * NoH / (4.0f * VoH);
//	}
//	else {
//		// Diffuse sample using cosine-weighted hemisphere
//		//xi.x = (xi.x - t) / (1.0 - t);
//		sampleDir = sampleCosineWeightedHemisphere(normal, xi);
//
//		pdf = max(dot(normal, sampleDir), 0.0) / PI;
//	}
//
//	return normalize(sampleDir);
//}

//__device__ float3 importanceSampleBRDF(float3 normal, float3 viewDir, const Material& material, uint32_t& seed, float& pdf, const SceneData& scene_data, float2 texture_uv) {
//	float roughness = material.Roughness;
//	float metallicity = material.Metallicity;
//	float3 H;
//	float3 sampleDir;
//
//	float random_value = randomFloat(seed);
//	float2 xi = make_float2(randomFloat(seed), randomFloat(seed));
//
//	float t = 0.5f;  // Threshold for deciding between diffuse and specular
//
//	if (random_value < t) {
//		// Diffuse
//		sampleDir = sampleCosineWeightedHemisphere(normal, xi);
//		pdf = dot(normal, sampleDir) * (1.0f / PI);
//	}
//	else {
//		// Specular
//		H = sampleGGX(normal, roughness, xi);
//		sampleDir = reflect(-viewDir, H);
//		float cosThetaH = dot(normal, H);
//		float cosThetaD = dot(sampleDir, H);
//		pdf = D_GGX(cosThetaH, roughness) * cosThetaH / (4.0f * cosThetaD);
//	}
//
//	return normalize(sampleDir);
//}