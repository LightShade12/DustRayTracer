#pragma once

class Mesh
{
public:
	Mesh() = default;

	char Name[32] = "unnamed";
	int* MaterialsIdx = nullptr;
	int MaterialCount = 0;
	int m_primitives_offset = -1;
	size_t m_trisCount = 0;
};