//******************************************
// APPLICATION ENTRY POINT
//******************************************

#include "core/Application/Application.hpp"
#include "core/Editor/EditorLayer.hpp"//include after application

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