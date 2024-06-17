#pragma once
#include "KeyCodes.hpp"
#include "glm/glm.hpp"
//TODO: add input abstraction layer

class Input
{
public:
	static glm::vec2 getMousePosition();
	static bool IsKeyDown(KeyCode keycode);
};