#include "core/engine.h"
#include "interface/viewport.h"
#include "tools/tools.h"
#define GLFW_INCLUDE_NONE

#include "callbacks.h"

#include <core/renderer.h>

#include <GLFW/glfw3.h>

// Moving this to a callback class would be cleaner so they can all share
// state instead of querying it every time needed
void Callbacks::mouse_callback(GLFWwindow *window, double xpos, double ypos) {
    auto& appState = GaussLabEngine::getAppState();

    Viewport::viewports[appState.selectedViewport]
        .view_camera->calculateDirection(window, xpos, ypos);
}

void Callbacks::scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    auto& appState = GaussLabEngine::getAppState();

    Viewport::viewports[appState.selectedViewport]
        .view_camera->calculateZoom(yoffset);

}

void Callbacks::key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    auto& appState = GaussLabEngine::getAppState();

    if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
        appState.in_view_mode = !(appState.in_view_mode);

    //TODO: recheck if necessary or if best method
    if (appState.in_view_mode)
        Tools::activeTool = Tools::AvailableTools::None;
}

