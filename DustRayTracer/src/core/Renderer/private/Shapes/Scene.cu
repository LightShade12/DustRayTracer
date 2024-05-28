#include "Scene.cuh"

//#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "core/Editor/Common/CudaCommon.cuh"

#include "core/Renderer/private/Kernel/BVH/BVHNode.cuh"

#include <cuda_runtime.h>
#include <thrust/host_vector.h>

#include <tiny_gltf.h>

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

bool Scene::loadMaterials(tinygltf::Model& model)
{
	//printf("loading materials\n\n");
	//printf("detected materials count in file: %d\n", model.materials.size());
	for (size_t matIdx = 0; matIdx < model.materials.size(); matIdx++)
	{
		tinygltf::Material mat = model.materials[matIdx];
		Material drt_mat;
		printf("material name: %s\n", mat.name.c_str());
		std::vector<double> color = mat.pbrMetallicRoughness.baseColorFactor;
		float3 albedo = { color[0], color[1], color[2] };//We just use RGB material albedo

		drt_mat.Albedo = albedo;
		drt_mat.AlbedoTextureIndex = mat.pbrMetallicRoughness.baseColorTexture.index;//should be -1 when empty
		printf("albedo texture idx: %d\n", drt_mat.AlbedoTextureIndex);
		m_Material.push_back(drt_mat);
	}
	//printf("loaded materials count: %d \n\n", m_Material.size());//should be 36 for cube

	return true;
}

bool Scene::loadTextures(tinygltf::Model& model, bool is_binary)
{
	const char* imgdir = "./src/models/";
	//printf("Images count in file: %d\n", model.images.size());
	//printf("total images: %zu\n", model.images.size());
	for (size_t textureIdx = 0; textureIdx < model.images.size(); textureIdx++)
	{
		tinygltf::Image current_img = model.images[textureIdx];
		//printf("image: %s\n", current_img.name.c_str());
		Texture tex;
		if (is_binary)
		{
			tinygltf::BufferView imgbufferview = model.bufferViews[current_img.bufferView];
			imgbufferview.byteOffset;
			imgbufferview.buffer;
			unsigned char* imgdata = model.buffers[imgbufferview.buffer].data.data() + imgbufferview.byteOffset;
			tex = Texture(imgdata, imgbufferview.byteLength);
		}
		else
		{
			//printf("uri: %s\n", current_img.uri.c_str());
			//printf("processing idx: %d,name= %s\n", textureIdx, current_img.name.c_str());
			//printf("load path: %s\n", (imgdir + current_img.uri).c_str());
			tex = Texture((imgdir + current_img.uri).c_str());
			//printf("img dims: w:%d h:%d ch:%d\n", tex.width, tex.height, tex.componentCount);
		}
		m_Textures.push_back(tex);//whitespace will be incorrectly parsed
	}
	return false;
}

//does not support reused mesh
bool parseMesh(tinygltf::Mesh mesh, tinygltf::Model model, std::vector<float3>& positions, std::vector<float3>& normals,
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
bool Scene::loadGLTFmodel(const char* filepath)
{
	bool isBinary = false;
	tinygltf::Model loadedmodel;
	loadModel(loadedmodel, filepath, isBinary);
	loadTextures(loadedmodel, isBinary);
	loadMaterials(loadedmodel);

	//mesh looping
	for (size_t nodeIdx = 0; nodeIdx < loadedmodel.nodes.size(); nodeIdx++)
	{
		std::vector<float3> loadedMeshPositions;
		std::vector<float3>loadedMeshNormals;
		std::vector<float2>loadedMeshUVs;
		std::vector<int>loadedMeshPrimitiveMatIdx;

		tinygltf::Node current_node = loadedmodel.nodes[nodeIdx];
		tinygltf::Mesh current_mesh = loadedmodel.meshes[current_node.mesh];

		//printf("processing node: %s with mesh: %s , mesh index= %d\n", current_node.name.c_str(), current_mesh.name.c_str(), current_node.mesh);
		Mesh loadedMesh;
		loadedMesh.m_primitives_offset = m_PrimitivesBuffer.size();

		parseMesh(current_mesh, loadedmodel, loadedMeshPositions,
			loadedMeshNormals, loadedMeshUVs, loadedMeshPrimitiveMatIdx);

		//printf("constructed positions count: %d \n", loadedMeshPositions.size());//should be 36 for cube
		//printf("constructed normals count: %d \n", loadedMeshNormals.size());//should be 36 for cube
		//printf("constructed UVs count: %d \n", loadedMeshUVs.size());//should be 36 for cube

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
			printf("positions-normals count mismatch!\n");
		}

		//Contruct Triangles
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

			////bounding box
			//for (size_t j = i; j < i + 3; j++)
			//{
			//	if (Bounds.pMax.x < positions[j].x)Bounds.pMax.x = positions[j].x;
			//	if (Bounds.pMax.y < positions[j].y)Bounds.pMax.y = positions[j].y;
			//	if (Bounds.pMax.z < positions[j].z)Bounds.pMax.z = positions[j].z;

			//	if (Bounds.pMin.x > positions[j].x)Bounds.pMin.x = positions[j].x;
			//	if (Bounds.pMin.y > positions[j].y)Bounds.pMin.y = positions[j].y;
			//	if (Bounds.pMin.z > positions[j].z)Bounds.pMin.z = positions[j].z;
			//}

			m_PrimitivesBuffer.push_back(Triangle(
				Vertex(loadedMeshPositions[i], loadedMeshNormals[i], loadedMeshUVs[i]),
				Vertex(loadedMeshPositions[i + 1], loadedMeshNormals[i + 1], loadedMeshUVs[i + 1]),
				Vertex(loadedMeshPositions[i + 2], loadedMeshNormals[i + 2], loadedMeshUVs[i + 2]),
				normalize(surface_normal),
				loadedMeshPrimitiveMatIdx[i / 3]));
		}

		loadedMesh.m_trisCount = m_PrimitivesBuffer.size() - loadedMesh.m_primitives_offset;

		//printf("constructing mesh\n");
		//printf("bbox max: x:%.3f y:%.3f z:%.3f \n", loadedMesh.Bounds.pMax.x, loadedMesh.Bounds.pMax.y, loadedMesh.Bounds.pMax.z);
		//printf("bbox min: x:%.3f y:%.3f z:%.3f \n", loadedMesh.Bounds.pMin.x, loadedMesh.Bounds.pMin.y, loadedMesh.Bounds.pMin.z);
		//printf("adding mesh\n");

		m_Meshes.push_back(loadedMesh);
		//printf("success\n\n");
	}

	return true;
};

Scene::~Scene()
{
	printf("freed scene\n");
	cudaDeviceSynchronize();

	if (d_BVHTreeRoot != nullptr)
	{
		d_BVHTreeRoot->Cleanup();
		cudaFree(d_BVHTreeRoot);
	}

	checkCudaErrors(cudaGetLastError());

	for (Texture texture : m_Textures)
	{
		texture.Cleanup();
	}
}