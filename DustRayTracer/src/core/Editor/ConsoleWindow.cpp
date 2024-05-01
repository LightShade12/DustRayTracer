#include "ConsoleWindow.hpp"

std::vector<const char*> ConsoleWindow::logs;

void ConsoleWindow::Render()
{
	ImGui::Begin("ConsoleWindow");
	for (const char* log : logs)
	{
		ImGui::Text(log);
	}

	ImGui::End();
}

void ConsoleWindow::logToConsole(const char* msg)
{
	logs.push_back(msg);
}