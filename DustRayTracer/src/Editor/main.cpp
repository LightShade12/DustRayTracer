//******************************************
// APPLICATION ENTRY POINT
//******************************************

#include "Application/Application.hpp"
#include "Editor/EditorLayer.hpp"//include after application

//------------------------------------------------------------------------g_
bool g_ApplicationRunning = true;

Application* CreateApplication(int argc, char** argv)
{
	ApplicationSpecification spec;
	spec.Name = "Window01";

	Application* app = new Application(spec);
	app->PushLayer<EditorLayer>();

	return app;
}


int main(int argc, char* argv[])
{
	while (g_ApplicationRunning)
	{
		Application* app = CreateApplication(argc, argv);
		app->Run();
		delete app;
	}

	return 0;

}

/*
* TODO: refactor utils and helpers into encapsulated logic
* TODO: add units to variables and arguments
* TODO: add Graphics/RenderAPI like API for rendering specific GPU stuff like creating: Textures, FrameBuffer, VertexBuffer, IndexBuffer, Pipelines, RenderPasses
* TODO: add RendererAPI for: PostFX, Camera, SceneGraph, Materials, Animation
*/