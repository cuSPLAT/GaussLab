#include "tools.h"
#include <imgui.h>
#include <iostream>
#include "meshslicing.h"
#include <GLFW/glfw3.h>

#include <core/renderer.h>

namespace Tools {

// For now we only have a slicing tool
AvailableTools activeTool = AvailableTools::None;

void dispatchToTool(GLFWwindow* window, int button, int action, int mod) {
    if (!(::globalState.windowHovered))
        return;

    if (button != GLFW_MOUSE_BUTTON_LEFT)
        return;

    switch (activeTool) {
        case AvailableTools::Slicing:
            MeshSlicing::captureMousePos(window, action);

        case AvailableTools::None:
        default:
            break;
    }
}

void drawToolBox_ImGui() {
    if (ImGui::Button("Slicing")) {
        activeTool = AvailableTools::Slicing;
        ::globalState.in_view_mode = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        const GLuint optional = glGetUniformLocation(
            ::globalState.vertexProgram, "planeExists");
        glUniform1i(optional, false);
    }
}

};
