#include "tools.h"
#include "meshslicing.h"
#include <GLFW/glfw3.h>

namespace Tools {

// For now we only have a slicing tool
AvailableTools activeTool = AvailableTools::Slicing;

void dispatchToTool(GLFWwindow* window, int button, int action, int mod) {
    if (button != GLFW_MOUSE_BUTTON_LEFT)
        return;

    switch (activeTool) {
        case AvailableTools::Slicing:
            MeshSlicing::captureMousePos(window);

        case AvailableTools::None:
        default:
            break;
    }
}

};
