#pragma once
//#include <vector_types.h>
#include <glm/glm.hpp>

struct Vertex
{
	Vertex() = default;
	Vertex(glm::vec3 pos, glm::vec3 nrm, glm::vec2 uv) :
		position({ pos.x,pos.y,pos.z }), normal({ nrm.x,nrm.y,nrm.z }), UV({ uv.x,uv.y }) {};
	float3 position;
	float3 normal;
	//float3 tangent;
	float2 UV;
};