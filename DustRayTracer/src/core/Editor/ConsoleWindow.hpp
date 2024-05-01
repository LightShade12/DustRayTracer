#pragma once
#include "core/Application/Application.hpp"

#include "imgui.h"

#include <vector>

class ConsoleWindow
{
public:
	ConsoleWindow() { logs = Application::Get().appLogs; };
	static void logToConsole(const char* msg);

	void Render();

private:
	static std::vector<const char*>logs;
};