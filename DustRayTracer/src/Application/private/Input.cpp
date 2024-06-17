#include "Input.hpp"
#include "Application/Application.hpp"

#include "GLFW/glfw3.h"

glm::vec2 Input::getMousePosition()
{
	double mouseX;
	double mouseY;
	glfwGetCursorPos(Application::Get().GetWindowHandle(), &mouseX, &mouseY);

	return glm::vec2(mouseX, mouseY);
}

bool Input::IsKeyDown(KeyCode keycode)
{
	GLFWwindow* windowHandle = Application::Get().GetWindowHandle();
	int state = glfwGetKey(windowHandle, (int)keycode);
	return state == GLFW_PRESS || state == GLFW_REPEAT;
}