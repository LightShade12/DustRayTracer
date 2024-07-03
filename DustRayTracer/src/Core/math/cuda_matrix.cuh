#include <vector_types.h>
#include "Core/math/cuda_math.cuh"

// This implementation follows the code from
// https://github.com/erwincoumans/experiments/blob/master/opencl/primitives/AdlPrimitives/Math/MathCL.h

#pragma once

/*****************************************
				Vector
/*****************************************/

__device__
inline float getSqrtf(float f2)
{
	return sqrtf(f2);
	//        return sqrt(f2);
}

__device__
inline float getReverseSqrt(float f2)
{
	return rsqrtf(f2);
}

__device__
inline float3 getCrossProduct(float3 a, float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__
inline float4 getCrossProduct(float4 a, float4 b)
{
	float3 v1 = make_float3(a.x, a.y, a.z);
	float3 v2 = make_float3(b.x, b.y, b.z);
	float3 v3 = make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);

	return make_float4(v3.x, v3.y, v3.z, 0.0f);
}

__device__
inline float getDotProduct(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__
inline float getDotProduct(float4 a, float4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ float3 getNormalizedVec(const float3 v)
{
	float invLen = 1.0f / sqrtf(getDotProduct(v, v));
	return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

__device__ float4 getNormalizedVec(const float4 v)
{
	float invLen = 1.0f / sqrtf(getDotProduct(v, v));
	return make_float4(v.x * invLen, v.y * invLen, v.z * invLen, v.w * invLen);
}

__device__
inline float dot3F4(float4 a, float4 b)
{
	float4 a1 = make_float4(a.x, a.y, a.z, 0.f);
	float4 b1 = make_float4(b.x, b.y, b.z, 0.f);
	return getDotProduct(a1, b1);
}

__device__
inline float getLength(float3 a)
{
	return sqrtf(getDotProduct(a, a));
}

__device__
inline float getLength(float4 a)
{
	return sqrtf(getDotProduct(a, a));
}

/*****************************************
				Matrix3x3
/*****************************************/

struct Matrix3x3_d
{
	__device__ Matrix3x3_d() = default;
	__device__ Matrix3x3_d(float3 r0, float3 r1, float3 r2) {
		m_row[0] = make_float4(r0); m_row[1] = make_float4(r1); m_row[2] = make_float4(r2);
	};
	float4 m_row[3];
	// Transpose function
	__device__ Matrix3x3_d transpose() const {
		Matrix3x3_d result;
		result.m_row[0].x = m_row[0].x; result.m_row[0].y = m_row[1].x; result.m_row[0].z = m_row[2].x;
		result.m_row[1].x = m_row[0].y; result.m_row[1].y = m_row[1].y; result.m_row[1].z = m_row[2].y;
		result.m_row[2].x = m_row[0].z; result.m_row[2].y = m_row[1].z; result.m_row[2].z = m_row[2].z;
		return result;
	}

	// Multiplication operator overload for float3
	__device__ float3 operator*(const float3& vec) const {
		float3 result;
		result.x = m_row[0].x * vec.x + m_row[0].y * vec.y + m_row[0].z * vec.z;
		result.y = m_row[1].x * vec.x + m_row[1].y * vec.y + m_row[1].z * vec.z;
		result.z = m_row[2].x * vec.x + m_row[2].y * vec.y + m_row[2].z * vec.z;
		return result;
	}
};

__device__
inline void setZero(Matrix3x3_d& m)
{
	m.m_row[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	m.m_row[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	m.m_row[2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__device__
inline Matrix3x3_d getZeroMatrix3x3()
{
	Matrix3x3_d m;
	m.m_row[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	m.m_row[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	m.m_row[2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	return m;
}

__device__
inline void setIdentity(Matrix3x3_d& m)
{
	m.m_row[0] = make_float4(1, 0, 0, 0);
	m.m_row[1] = make_float4(0, 1, 0, 0);
	m.m_row[2] = make_float4(0, 0, 1, 0);
}

__device__
inline Matrix3x3_d getIdentityMatrix3x3()
{
	Matrix3x3_d m;
	m.m_row[0] = make_float4(1, 0, 0, 0);
	m.m_row[1] = make_float4(0, 1, 0, 0);
	m.m_row[2] = make_float4(0, 0, 1, 0);
	return m;
}

__device__
inline Matrix3x3_d getTranspose(const Matrix3x3_d m)
{
	Matrix3x3_d out;
	out.m_row[0] = make_float4(m.m_row[0].x, m.m_row[1].x, m.m_row[2].x, 0.f);
	out.m_row[1] = make_float4(m.m_row[0].y, m.m_row[1].y, m.m_row[2].y, 0.f);
	out.m_row[2] = make_float4(m.m_row[0].z, m.m_row[1].z, m.m_row[2].z, 0.f);
	return out;
}

__device__
inline Matrix3x3_d MatrixMul(Matrix3x3_d& a, Matrix3x3_d& b)
{
	Matrix3x3_d transB = getTranspose(b);
	Matrix3x3_d ans;
	//        why this doesn't run when 0ing in the for{}
	a.m_row[0].w = 0.f;
	a.m_row[1].w = 0.f;
	a.m_row[2].w = 0.f;
	for (int i = 0; i < 3; i++)
	{
		//        a.m_row[i].w = 0.f;
		ans.m_row[i].x = dot3F4(a.m_row[i], transB.m_row[0]);
		ans.m_row[i].y = dot3F4(a.m_row[i], transB.m_row[1]);
		ans.m_row[i].z = dot3F4(a.m_row[i], transB.m_row[2]);
		ans.m_row[i].w = 0.f;
	}
	return ans;
}

/*****************************************
				Quaternion
/*****************************************/

typedef float4 Quaternion;

__device__
inline Quaternion quaternionMul(Quaternion a, Quaternion b);

__device__
inline Quaternion qtNormalize(Quaternion in);

__device__
inline float4 qtRotate(Quaternion q, float4 vec);

__device__
inline Quaternion qtInvert(Quaternion q);

__device__
inline Matrix3x3_d qtGetRotationMatrix(Quaternion q);

__device__
inline Quaternion quaternionMul(Quaternion a, Quaternion b)
{
	Quaternion ans;
	ans = getCrossProduct(a, b);
	ans = make_float4(ans.x + a.w * b.x + b.w * a.x + b.w * a.y, ans.y + a.w * b.y + b.w * a.z, ans.z + a.w * b.z, ans.w + a.w * b.w + b.w * a.w);
	//        ans.w = a.w*b.w - (a.x*b.x+a.y*b.y+a.z*b.z);
	ans.w = a.w * b.w - dot3F4(a, b);
	return ans;
}

__device__
inline Quaternion qtNormalize(Quaternion in)
{
	return getNormalizedVec(in);
	//        in /= length( in );
	//        return in;
}

__device__
inline Quaternion qtInvert(const Quaternion q)
{
	return make_float4(-q.x, -q.y, -q.z, q.w);
}

__device__
inline float4 qtRotate(const Quaternion q, const float4 vec)
{
	Quaternion qInv = qtInvert(q);
	float4 vcpy = vec;
	vcpy.w = 0.f;
	float4 out = quaternionMul(quaternionMul(q, vcpy), qInv);
	return out;
}

__device__
inline float4 qtInvRotate(const Quaternion q, const float4 vec)
{
	return qtRotate(qtInvert(q), vec);
}

__device__
inline Matrix3x3_d qtGetRotationMatrix(Quaternion quat)
{
	float4 quat2 = make_float4(quat.x * quat.x, quat.y * quat.y, quat.z * quat.z, 0.f);
	Matrix3x3_d out;

	out.m_row[0].x = 1 - 2 * quat2.y - 2 * quat2.z;
	out.m_row[0].y = 2 * quat.x * quat.y - 2 * quat.w * quat.z;
	out.m_row[0].z = 2 * quat.x * quat.z + 2 * quat.w * quat.y;
	out.m_row[0].w = 0.f;

	out.m_row[1].x = 2 * quat.x * quat.y + 2 * quat.w * quat.z;
	out.m_row[1].y = 1 - 2 * quat2.x - 2 * quat2.z;
	out.m_row[1].z = 2 * quat.y * quat.z - 2 * quat.w * quat.x;
	out.m_row[1].w = 0.f;

	out.m_row[2].x = 2 * quat.x * quat.z - 2 * quat.w * quat.y;
	out.m_row[2].y = 2 * quat.y * quat.z + 2 * quat.w * quat.x;
	out.m_row[2].z = 1 - 2 * quat2.x - 2 * quat2.y;
	out.m_row[2].w = 0.f;

	return out;
}