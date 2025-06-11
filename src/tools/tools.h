#pragma once

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace Tools {

enum class AvailableTools {
    None,
    Slicing
};

extern AvailableTools activeTool;

void dispatchToTool(GLFWwindow* window, int button, int action, int mod);

void drawToolBox_ImGui();
};
