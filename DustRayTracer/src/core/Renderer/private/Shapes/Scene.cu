#include "Scene.cuh"

#include <tiny_gltf.h>

bool loadModel(tinygltf::Model& model, const char* filename) {
	tinygltf::TinyGLTF loader;
	std::string err;
	std::string warn;

	bool res = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
	if (!warn.empty()) {
		std::cout << "WARN: " << warn << std::endl;
	}

	if (!err.empty()) {
		std::cout << "ERR: " << err << std::endl;
	}

	if (!res)
		std::cout << "Failed to load glTF: " << filename << std::endl;
	else
		std::cout << "Loaded glTF: " << filename << std::endl;

	return res;
}

bool Scene::loadMaterials(tinygltf::Model& model)
{
	printf("loading materials\n\n");
	printf("detected materials count in file: %d\n", model.materials.size());
	for (size_t matIdx = 0; matIdx < model.materials.size(); matIdx++)
	{
		tinygltf::Material mat = model.materials[matIdx];
		std::vector<double> color = mat.pbrMetallicRoughness.baseColorFactor;
		float3 albedo = { color[0], color[1], color[2] };//We just use RGB material albedo
		m_Material.push_back(Material(albedo));
	}
	printf("loaded materials count: %d \n\n", m_Material.size());//should be 36 for cube

	return true;
}

bool parseMesh(tinygltf::Mesh mesh, tinygltf::Model model, std::vector<float3>& positions, std::vector<float3>& normals)
{
	for (size_t primIdx = 0; primIdx < mesh.primitives.size(); primIdx++)
	{
		int pos_attrib_accesorIdx = mesh.primitives[primIdx].attributes["POSITION"];
		int nrm_attrib_accesorIdx = mesh.primitives[primIdx].attributes["NORMAL"];
		int indices_accesorIdx = mesh.primitives[primIdx].indices;

		tinygltf::Accessor pos_accesor = model.accessors[pos_attrib_accesorIdx];
		tinygltf::Accessor nrm_accesor = model.accessors[nrm_attrib_accesorIdx];
		tinygltf::Accessor indices_accesor = model.accessors[indices_accesorIdx];

		int pos_accesor_byte_offset = pos_accesor.byteOffset;//redundant
		int nrm_accesor_byte_offset = nrm_accesor.byteOffset;//redundant
		int indices_accesor_byte_offset = indices_accesor.byteOffset;//redundant

		tinygltf::BufferView pos_bufferview = model.bufferViews[pos_accesor.bufferView];
		tinygltf::BufferView nrm_bufferview = model.bufferViews[nrm_accesor.bufferView];
		tinygltf::BufferView indices_bufferview = model.bufferViews[indices_accesor.bufferView];

		int pos_buffer_byte_offset = pos_bufferview.byteOffset;
		int nrm_buffer_byte_offset = nrm_bufferview.byteOffset;

		tinygltf::Buffer indices_buffer = model.buffers[indices_bufferview.buffer];//should alawys be zero?

		printf("normals accesor count: %d\n", nrm_accesor.count);
		printf("positions accesor count: %d\n", pos_accesor.count);
		printf("indices accesor count: %d\n", indices_accesor.count);

		unsigned short* indicesbuffer = (unsigned short*)(indices_buffer.data.data());
		float3* positions_buffer = (float3*)(indices_buffer.data.data() + pos_buffer_byte_offset);
		float3* normals_buffer = (float3*)(indices_buffer.data.data() + nrm_buffer_byte_offset);

		for (int i = (indices_bufferview.byteOffset / 2); i < (indices_bufferview.byteLength + indices_bufferview.byteOffset) / 2; i++)
		{
			positions.push_back(positions_buffer[indicesbuffer[i]]);
			normals.push_back(normals_buffer[indicesbuffer[i]]);
		}
	}
	return true;
}

//tinyGLTF impl
bool Scene::loadGLTFmodel(const char* filepath)
{
	tinygltf::Model loadedmodel;
	loadModel(loadedmodel, filepath);
	loadMaterials(loadedmodel);

	//mesh looping
	for (size_t meshIdx = 0; meshIdx < loadedmodel.meshes.size(); meshIdx++)
	{
		std::vector<float3> loadedMeshPositions;
		std::vector<float3>loadedMeshNormals;

		tinygltf::Mesh current_mesh = loadedmodel.meshes[meshIdx];

		printf("processing mesh: %s , index= %d\n", current_mesh.name, meshIdx);

		parseMesh(current_mesh, loadedmodel, loadedMeshPositions, loadedMeshNormals);

		printf("constructed positions count: %d \n", loadedMeshPositions.size());//should be 36 for cube
		printf("constructed normals count: %d \n", loadedMeshNormals.size());//should be 36 for cube

		//DEBUG positions-normal-data print
		if (loadedMeshPositions.size() == loadedMeshNormals.size())
		{
			bool stop = false;
			printf("positions:\n");
			for (size_t i = 0; i < loadedMeshPositions.size(); i++)
			{
				if (i > 3 && i < loadedMeshPositions.size() - 3)
				{
					if (!stop)
					{
						printf("...\n");
						stop = true;
					}
					continue;
				}
				float3 pos = loadedMeshPositions[i];
				printf("x:%.3f y:%.3f z:%.3f\n", pos.x, pos.y, pos.z);
			}
			stop = false;
			printf("normals:\n");
			for (size_t i = 0; i < loadedMeshNormals.size(); i++)
			{
				if (i > 3 && i < loadedMeshNormals.size() - 3)
				{
					if (!stop)
					{
						printf("...\n");
						stop = true;
					}
					continue;
				}
				float3 nrm = loadedMeshNormals[i];
				printf("x:%.3f y:%.3f z:%.3f\n", nrm.x, nrm.y, nrm.z);
			}
		}
		else
		{
			printf("positions-normals count mismatch!\n");
		}

		printf("constructing mesh\n");
		Mesh loadedMesh(loadedMeshPositions, loadedMeshNormals, current_mesh.primitives[0].material);//TODO: does not support per primitive material so idx=0 for now
		printf("adding mesh\n");
		m_Meshes.push_back(loadedMesh);
		printf("success\n\n");
	}

	return true;
};

Scene::~Scene()
{
	for (Mesh mesh : m_Meshes)
	{
		mesh.Cleanup();
	}
}