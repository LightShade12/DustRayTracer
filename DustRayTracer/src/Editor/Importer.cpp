#include "Importer.h"
#include "Common/dbg_macros.hpp"

#include <tiny_gltf.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include <iostream>

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

bool Importer::loadMaterials(const tinygltf::Model& model)
{
	printToConsole("detected materials count in file: %zu\n", model.materials.size());
	for (size_t matIdx = 0; matIdx < model.materials.size(); matIdx++)
	{
		DustRayTracer::HostMaterial drt_material;
		tinygltf::Material gltf_material = model.materials[matIdx];
		printToConsole("loading material: %s\n", gltf_material.name.c_str());
		tinygltf::PbrMetallicRoughness PBR_data = gltf_material.pbrMetallicRoughness;
		memset(drt_material.getNamePtr(), 0, sizeof(drt_material.getName()));
		strncpy(drt_material.getNamePtr(), gltf_material.name.c_str(), gltf_material.name.size());
		drt_material.getNamePtr()[gltf_material.name.size()] = '\0';
		drt_material.setAlbedo(make_float3(PBR_data.baseColorFactor[0], PBR_data.baseColorFactor[1], PBR_data.baseColorFactor[2]));//We just use RGB material albedo for now
		drt_material.setEmissiveColor(make_float3(gltf_material.emissiveFactor[0], gltf_material.emissiveFactor[1], gltf_material.emissiveFactor[2]));
		if (PBR_data.baseColorTexture.index >= 0)drt_material.setAlbedoTextureIndex(model.textures[PBR_data.baseColorTexture.index].source);
		if (PBR_data.metallicRoughnessTexture.index >= 0)drt_material.setRoughnessTextureIndex(model.textures[PBR_data.metallicRoughnessTexture.index].source);
		if (gltf_material.normalTexture.index >= 0)drt_material.setNormalTextureIndex(model.textures[gltf_material.normalTexture.index].source);
		drt_material.setNormalMapScale(gltf_material.normalTexture.scale);
		if (gltf_material.emissiveTexture.index >= 0)drt_material.setEmissionTextureIndex(model.textures[gltf_material.emissiveTexture.index].source);
		drt_material.setMetallicity((PBR_data.metallicRoughnessTexture.index >= 0) ? 1 : PBR_data.metallicFactor);
		drt_material.setRoughness((PBR_data.metallicRoughnessTexture.index >= 0) ? 0.6f : PBR_data.roughnessFactor);
		//printToConsole("albedo texture idx: %d\n", drt_material.AlbedoTextureIndex);
		m_WorkingScene->addMaterial(drt_material);
	}
	//printf("loaded materials count: %d \n\n", m_MaterialsBuffer.size());

	return true;
}

//TODO: probably already handled by tinygltf stb; redundant stb call?
bool Importer::loadTextures(const tinygltf::Model& model, bool is_binary)
{
	const char* image_reference_directory = "../models/";
	printToConsole("detected images count in file: %zu\n", model.images.size());

	for (size_t texture_idx = 0; texture_idx < model.images.size(); texture_idx++)
	{
		tinygltf::Image gltf_image = model.images[texture_idx];
		printToConsole("loading image: %s\n", gltf_image.name.c_str());
		Texture drt_texture;
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
		if (drt_texture.d_data == nullptr) {
			printToConsole("Image data null\n"); return false;
		}
		memset(drt_texture.Name, 0, sizeof(drt_texture.Name));
		strncpy(drt_texture.Name, gltf_image.name.c_str(), gltf_image.name.size());
		drt_texture.Name[gltf_image.name.size()] = '\0';
		drt_texture.ChannelBitDepth = gltf_image.bits;
		m_WorkingScene->addTexture(drt_texture);//whitespace will be incorrectly parsed
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
bool Importer::loadGLTF(const char* filepath, DustRayTracer::HostScene& scene_object)
{
	m_WorkingScene = &scene_object;
	bool is_binary = false;
	tinygltf::Model loadedmodel;
	if (!loadModel(loadedmodel, filepath, is_binary)) return false;
	loadTextures(loadedmodel, is_binary);
	loadMaterials(loadedmodel);

#ifdef DEBUG
	for (std::string extensionname : loadedmodel.extensionsUsed) {
		printf("using: %s\n", extensionname.c_str());
	}
	for (std::string extensionname : loadedmodel.extensionsRequired) {
		printf("required: %s\n", extensionname.c_str());
	}
#endif // DEBUG

	printToConsole("Detected nodes in file:%zu\n", loadedmodel.nodes.size());
	printToConsole("Detected meshes in file:%zu\n", loadedmodel.meshes.size());
	printToConsole("Detected cameras in file:%zu\n", loadedmodel.cameras.size());

	//node looping
	for (size_t nodeIdx = 0; nodeIdx < loadedmodel.nodes.size(); nodeIdx++)
	{
		std::vector<float3> loadedMeshPositions;
		std::vector<float3>loadedMeshNormals;
		std::vector<float2>loadedMeshUVs;
		std::vector<int>loadedMeshPrimitiveMatIdx;

		tinygltf::Node gltf_node = loadedmodel.nodes[nodeIdx];
		printToConsole("Processing node: %s\n", gltf_node.name.c_str());

		if (gltf_node.camera >= 0) {
			tinygltf::Camera gltf_camera = loadedmodel.cameras[gltf_node.camera];
			printToConsole("\nfound a camera: %s\n", gltf_camera.name.c_str());
			float3 cpos = { gltf_node.translation[0] ,gltf_node.translation[1] ,gltf_node.translation[2] };
			DustRayTracer::HostCamera drt_camera;
			memset(drt_camera.getNamePtr(), 0, sizeof(drt_camera.getName()));
			strncpy(drt_camera.getNamePtr(), gltf_camera.name.c_str(), gltf_camera.name.size());
			drt_camera.getNamePtr()[gltf_camera.name.size()] = '\0';

			drt_camera.setPosition(glm::vec3(cpos.x, cpos.y, cpos.z));
			drt_camera.setVerticalFOV(gltf_camera.perspective.yfov);

			if (gltf_node.rotation.size() > 0) {
				float qx = gltf_node.rotation[0];
				float qy = gltf_node.rotation[1];
				float qz = gltf_node.rotation[2];
				float qw = gltf_node.rotation[3];
				glm::quat quaternion(qw, qx, qy, qz);
				glm::mat4 rotationMatrix = glm::toMat4(quaternion);
				glm::vec3 forwardDir = -glm::vec3(rotationMatrix[2]);
				float3 lookDir = make_float3(forwardDir.x, forwardDir.y, forwardDir.z);
				drt_camera.setLookDir(glm::vec3(lookDir.x, lookDir.y, lookDir.z));
			}

			m_WorkingScene->addCamera(drt_camera);
		}
		if (gltf_node.mesh < 0)continue;//TODO: crude fix
		tinygltf::Mesh gltf_mesh = loadedmodel.meshes[gltf_node.mesh];

		Mesh drt_mesh;
		memset(drt_mesh.Name, 0, sizeof(drt_mesh.Name));
		strncpy(drt_mesh.Name, gltf_mesh.name.c_str(), gltf_mesh.name.size());
		drt_mesh.Name[gltf_mesh.name.size()] = '\0';
		printToConsole("\nprocessing mesh:%s\n", gltf_mesh.name.c_str());

		drt_mesh.m_primitives_offset = m_WorkingScene->getTrianglesBufferSize();

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

			uint32_t mtidx = loadedMeshPrimitiveMatIdx[i / 3];

			m_WorkingScene->addTriangle(Triangle(
				Vertex(loadedMeshPositions[i], loadedMeshNormals[i], loadedMeshUVs[i]),
				Vertex(loadedMeshPositions[i + 1], loadedMeshNormals[i + 1], loadedMeshUVs[i + 1]),
				Vertex(loadedMeshPositions[i + 2], loadedMeshNormals[i + 2], loadedMeshUVs[i + 2]),
				normalize(surface_normal),
				mtidx));
			DustRayTracer::HostMaterial mat = m_WorkingScene->getMaterial(mtidx);
			float3 emcol = mat.getEmissiveColor();
			if (mat.getEmissionTextureIndex() >= 0 ||
				!(emcol.x == 0 && emcol.y == 0 && emcol.z == 0)) {
				m_WorkingScene->addTriangleLightidx(m_WorkingScene->getTrianglesBufferSize() - 1);
			}
		}

		drt_mesh.m_trisCount = m_WorkingScene->getTrianglesBufferSize() - drt_mesh.m_primitives_offset;

		m_WorkingScene->addMesh(drt_mesh);
		printToConsole("\rloaded mesh:%zu/%zu", nodeIdx + 1, loadedmodel.nodes.size());
	}

	printf("mesh lights %zu \n", m_WorkingScene->getTriangleLightsBufferSize());

	//construct default camera
	if (m_WorkingScene->getCamerasBufferSize() == 0)
	{
		DustRayTracer::HostCamera default_camera;
		m_WorkingScene->addCamera(default_camera);
	}
	//printToConsole("meshlights:%zu\n", m_TriangleLightsIndicesBuffer.size());
	printToConsole("\n");

	return true;
}