#include "Scene.cuh"

Scene::~Scene()
{
	for (Mesh mesh : m_Meshes)
	{
		mesh.Cleanup();
	}
}