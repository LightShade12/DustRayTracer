#include "Scene.cuh"
//TODO: apparently this must exist on the device: its the standard
//#define STB_IMAGE_IMPLEMENTATION
#include "Camera.cuh"
#include "Common/dbg_macros.hpp"
#include "Editor/Common/CudaCommon.cuh"

#include "Core/BVH/BVHNode.cuh"

#include "stb_image.h"
#include <tiny_gltf.h>

#include <thrust/host_vector.h>
//TODO: texture reuse doesnt work
static std::string GetFilePathExtension(const std::string& FileName) {
	if (FileName.find_last_of(".") != std::string::npos)
		return FileName.substr(FileName.find_last_of(".") + 1);
	return "";
}

bool loadModel(tinygltf::Model& model, const char* filename, bool& is_binary) {
	tinygltf::TinyGLTF loader;
	std::string err;
	std::string warn;

	std::string ext = GetFilePathExtension(filename);

	//printf("extension %s \n", ext.c_str());

	bool res = false;
	if (ext.compare("glb") == 0) {
		// assume binary glTF.
		res =
			loader.LoadBinaryFromFile(&model, &err, &warn, filename);
		is_binary = true;
	}
	else {
		// assume ascii glTF.
		res = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
	}

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

bool Scene::loadMaterials(const tinygltf::Model& model)
{
	printToConsole("detected materials count in file: %d\n", model.materials.size());
	for (size_t matIdx = 0; matIdx < model.materials.size(); matIdx++)
	{
		Material drt_material;
		tinygltf::Material gltf_material = model.materials[matIdx];
		printToConsole("material name: %s\n", gltf_material.name.c_str());
		tinygltf::PbrMetallicRoughness PBR_data = gltf_material.pbrMetallicRoughness;
		memset(drt_material.Name, 0, sizeof(drt_material.Name));
		strncpy(drt_material.Name, gltf_material.name.c_str(), gltf_material.name.size());
		drt_material.Name[gltf_material.name.size()] = '\0';
		drt_material.Albedo = make_float3(PBR_data.baseColorFactor[0], PBR_data.baseColorFactor[1], PBR_data.baseColorFactor[2]);//We just use RGB material albedo for now
		drt_material.EmissiveColor = make_float3(gltf_material.emissiveFactor[0], gltf_material.emissiveFactor[1], gltf_material.emissiveFactor[2]);
		drt_material.AlbedoTextureIndex = PBR_data.baseColorTexture.index;//should be -1 when empty
		drt_material.RoughnessTextureIndex = PBR_data.metallicRoughnessTexture.index;
		drt_material.NormalTextureIndex = gltf_material.normalTexture.index;
		drt_material.NormalMapScale = gltf_material.normalTexture.scale;
		drt_material.EmissionTextureIndex = gltf_material.emissiveTexture.index;
		drt_material.Metallicity = PBR_data.metallicFactor;
		drt_material.Roughness = PBR_data.roughnessFactor;
		//printToConsole("albedo texture idx: %d\n", drt_material.AlbedoTextureIndex);
		m_Materials.push_back(drt_material);
	}
	//printf("loaded materials count: %d \n\n", m_Materials.size());

	return true;
}

//TODO: probably already handled by tinygltf stb; redundant stb call?
bool Scene::loadTextures(const tinygltf::Model& model, bool is_binary)
{
	const char* image_reference_directory = "../models/";
	printToConsole("detected images count in file: %zu\n", model.images.size());

	for (size_t texture_idx = 0; texture_idx < model.images.size(); texture_idx++)
	{
		tinygltf::Image gltf_image = model.images[texture_idx];
		Texture drt_texture;
		memset(drt_texture.Name, 0, sizeof(drt_texture.Name));
		strncpy(drt_texture.Name, gltf_image.name.c_str(), gltf_image.name.size());
		drt_texture.Name[gltf_image.name.size()] = '\0';
		drt_texture.ChannelBitDepth = gltf_image.bits;

		if (is_binary)
		{
			tinygltf::BufferView imgbufferview = model.bufferViews[gltf_image.bufferView];
			const unsigned char* imgdata = model.buffers[imgbufferview.buffer].data.data() + imgbufferview.byteOffset;
			drt_texture = Texture(imgdata, imgbufferview.byteLength);
		}
		else
		{
			drt_texture = Texture((image_reference_directory + gltf_image.uri).c_str());
		}
		if (drt_texture.d_data == nullptr)return false;
		m_Textures.push_back(drt_texture);//whitespace will be incorrectly parsed
	}
	return true;
}

//does not support reused mesh
static bool parseMesh(tinygltf::Mesh mesh, tinygltf::Model model, std::vector<float3>& positions, std::vector<float3>& normals,
	std::vector<float2>& UVs, std::vector<int>& prim_mat_idx)
{
	//printf("total primitives: %zu\n", mesh.primitives.size());
	for (size_t primIdx = 0; primIdx < mesh.primitives.size(); primIdx++)
	{
		//printf("prim idx:%zu \n", primIdx);
		tinygltf::Primitive primitive = mesh.primitives[primIdx];

		int pos_attrib_accesorIdx = primitive.attributes["POSITION"];
		int nrm_attrib_accesorIdx = primitive.attributes["NORMAL"];
		int uv_attrib_accesorIdx = primitive.attributes["TEXCOORD_0"];

		int indices_accesorIdx = primitive.indices;

		tinygltf::Accessor pos_accesor = model.accessors[pos_attrib_accesorIdx];
		tinygltf::Accessor nrm_accesor = model.accessors[nrm_attrib_accesorIdx];
		tinygltf::Accessor uv_accesor = model.accessors[uv_attrib_accesorIdx];
		tinygltf::Accessor indices_accesor = model.accessors[indices_accesorIdx];

		int pos_accesor_byte_offset = pos_accesor.byteOffset;//redundant
		int nrm_accesor_byte_offset = nrm_accesor.byteOffset;//redundant
		int uv_accesor_byte_offset = uv_accesor.byteOffset;//redundant
		int indices_accesor_byte_offset = indices_accesor.byteOffset;//redundant

		tinygltf::BufferView pos_bufferview = model.bufferViews[pos_accesor.bufferView];
		tinygltf::BufferView nrm_bufferview = model.bufferViews[nrm_accesor.bufferView];
		tinygltf::BufferView uv_bufferview = model.bufferViews[uv_accesor.bufferView];
		tinygltf::BufferView indices_bufferview = model.bufferViews[indices_accesor.bufferView];

		int pos_buffer_byte_offset = pos_bufferview.byteOffset;
		int nrm_buffer_byte_offset = nrm_bufferview.byteOffset;
		int uv_buffer_byte_offset = uv_bufferview.byteOffset;

		tinygltf::Buffer indices_buffer = model.buffers[indices_bufferview.buffer];//should alawys be zero?

		//printf("normals accesor count: %d\n", nrm_accesor.count);
		//printf("positions accesor count: %d\n", pos_accesor.count);
		//printf("UVs accesor count: %d\n", uv_accesor.count);
		//printf("indices accesor count: %d\n", indices_accesor.count);

		unsigned short* indicesbuffer = (unsigned short*)(indices_buffer.data.data());
		float3* positions_buffer = (float3*)(indices_buffer.data.data() + pos_buffer_byte_offset);
		float3* normals_buffer = (float3*)(indices_buffer.data.data() + nrm_buffer_byte_offset);
		float2* UVs_buffer = (float2*)(indices_buffer.data.data() + uv_buffer_byte_offset);

		for (int i = (indices_bufferview.byteOffset / 2); i < (indices_bufferview.byteLength + indices_bufferview.byteOffset) / 2; i++)
		{
			positions.push_back(positions_buffer[indicesbuffer[i]]);
			normals.push_back(normals_buffer[indicesbuffer[i]]);
			UVs.push_back(UVs_buffer[indicesbuffer[i]]);
		}
		for (size_t i = 0; i < indices_accesor.count / 3; i++)//no of triangles per primitive
		{
			prim_mat_idx.push_back(primitive.material);
		}
	}
	return true;
}

//tinyGLTF impl
bool Scene::loadGLTFmodel(const char* filepath, Camera** camera)
{
	bool is_binary = false;
	tinygltf::Model loadedmodel;
	loadModel(loadedmodel, filepath, is_binary);
	loadTextures(loadedmodel, is_binary);
	loadMaterials(loadedmodel);

	printToConsole("Detected mesh count in file:%zu\n", loadedmodel.meshes.size());

	//mesh looping
	for (size_t nodeIdx = 0; nodeIdx < loadedmodel.nodes.size(); nodeIdx++)
	{
		std::vector<float3> loadedMeshPositions;
		std::vector<float3>loadedMeshNormals;
		std::vector<float2>loadedMeshUVs;
		std::vector<int>loadedMeshPrimitiveMatIdx;

		tinygltf::Node gltf_node = loadedmodel.nodes[nodeIdx];
		if (gltf_node.camera >= 0) {
			printToConsole("found camera\n");
			tinygltf::Camera gltf_camera = loadedmodel.cameras[gltf_node.camera];
			float3 cpos = { gltf_node.translation[0] ,gltf_node.translation[1] ,gltf_node.translation[2] };
			*camera = new Camera(cpos);
			(*camera)->vfov_rad = gltf_camera.perspective.yfov;
		}
		if (gltf_node.mesh < 0)continue;//TODO: crude fix
		tinygltf::Mesh gltf_mesh = loadedmodel.meshes[gltf_node.mesh];

		Mesh drt_mesh;
		memset(drt_mesh.Name, 0, sizeof(drt_mesh.Name));
		strncpy(drt_mesh.Name, gltf_mesh.name.c_str(), gltf_mesh.name.size());
		drt_mesh.Name[gltf_mesh.name.size()] = '\0';
		printToConsole("\nprocessing mesh:%s\n", gltf_mesh.name.c_str());

		drt_mesh.m_primitives_offset = m_PrimitivesBuffer.size();

		parseMesh(gltf_mesh, loadedmodel, loadedMeshPositions,
			loadedMeshNormals, loadedMeshUVs, loadedMeshPrimitiveMatIdx);

		//DEBUG positions-normal-d_data print
		if (loadedMeshPositions.size() == loadedMeshNormals.size())
		{
			bool stop = false;
			//printf("positions:\n");
			for (size_t i = 0; i < loadedMeshPositions.size(); i++)
			{
				if (i > 2 && i < loadedMeshPositions.size() - 3)
				{
					if (!stop)
					{
						//printf("...\n");
						stop = true;
					}
					continue;
				}
				float3 pos = loadedMeshPositions[i];
				//printf("x:%.3f y:%.3f z:%.3f\n", pos.x, pos.y, pos.z);
			}
			stop = false;
			//printf("normals:\n");
			for (size_t i = 0; i < loadedMeshNormals.size(); i++)
			{
				if (i > 2 && i < loadedMeshNormals.size() - 3)
				{
					if (!stop)
					{
						//printf("...\n");
						stop = true;
					}
					continue;
				}
				float3 nrm = loadedMeshNormals[i];
				//printf("x:%.3f y:%.3f z:%.3f\n", nrm.x, nrm.y, nrm.z);
			}
			stop = false;
			//printf("UVs:\n");
			for (size_t i = 0; i < loadedMeshUVs.size(); i++)
			{
				if (i > 2 && i < loadedMeshUVs.size() - 3)
				{
					if (!stop)
					{
						//printf("...\n");
						stop = true;
					}
					continue;
				}
				float2 uv = loadedMeshUVs[i];
				//printf("U:%.3f V:%.3f \n", uv.x, uv.y);
			}
		}
		else
		{
			printToConsole("positions-normals count mismatch!\n");
		}

		//Contruct and push Triangles
		//Positions.size() and vertex_normals.size() must be equal!
		for (size_t i = 0; i < loadedMeshPositions.size(); i += 3)
		{
			//surface normal construction
			float3 p0 = loadedMeshPositions[i + 1] - loadedMeshPositions[i];
			float3 p1 = loadedMeshPositions[i + 2] - loadedMeshPositions[i];
			float3 faceNormal = cross(p0, p1);

			float3 avgVertexNormal = (loadedMeshNormals[i] + loadedMeshNormals[i + 1] + loadedMeshNormals[i + 2]) / 3;
			float ndot = dot(faceNormal, avgVertexNormal);

			float3 surface_normal = (ndot < 0.0f) ? -faceNormal : faceNormal;

			m_PrimitivesBuffer.push_back(Triangle(
				Vertex(loadedMeshPositions[i], loadedMeshNormals[i], loadedMeshUVs[i]),
				Vertex(loadedMeshPositions[i + 1], loadedMeshNormals[i + 1], loadedMeshUVs[i + 1]),
				Vertex(loadedMeshPositions[i + 2], loadedMeshNormals[i + 2], loadedMeshUVs[i + 2]),
				normalize(surface_normal),
				loadedMeshPrimitiveMatIdx[i / 3]));

			/*if (m_Materials[loadedMeshPrimitiveMatIdx[i / 3]].EmissionTextureIndex >= 0 ||
				!(m_Materials[loadedMeshPrimitiveMatIdx[i / 3]].EmissiveColor.x == 0 &&
					m_Materials[loadedMeshPrimitiveMatIdx[i / 3]].EmissiveColor.y == 0 &&
					m_Materials[loadedMeshPrimitiveMatIdx[i / 3]].EmissiveColor.z == 0)) {
				m_MeshLights.push_back(m_PrimitivesBuffer.size() - 1);
			}*/
		}

		drt_mesh.m_trisCount = m_PrimitivesBuffer.size() - drt_mesh.m_primitives_offset;

		m_Meshes.push_back(drt_mesh);
		printToConsole("\rloaded mesh:%zu/%zu", nodeIdx + 1, loadedmodel.nodes.size());
	}

	//printToConsole("meshlights:%zu\n", m_MeshLights.size());
	printToConsole("\n");

	return true;
};

Scene::~Scene()
{
	cudaDeviceSynchronize();

	thrust::host_vector<BVHNode>nodes = m_BVHNodes;

	for (BVHNode node : nodes) {
		//printf("node freed\n");
		node.Cleanup();
	}

	checkCudaErrors(cudaGetLastError());

	for (Texture texture : m_Textures)
	{
		texture.Cleanup();
	}
	checkCudaErrors(cudaGetLastError());

	printf("freed scene\n");
}