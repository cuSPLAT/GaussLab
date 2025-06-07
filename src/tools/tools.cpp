#include "tools.h"
#include <iostream>
#include "meshslicing.h"
#include <GLFW/glfw3.h>

#include <core/renderer.h>

namespace Tools {

// For now we only have a slicing tool
AvailableTools activeTool = AvailableTools::Slicing;

void dispatchToTool(GLFWwindow* window, int button, int action, int mod) {
    std::cout << ::globalState.selectedViewport << std::endl;
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

};
