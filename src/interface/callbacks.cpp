#include "interface/viewport.h"
#include "tools/tools.h"
#define GLFW_INCLUDE_NONE

#include "callbacks.h"

#include <core/renderer.h>

#include <GLFW/glfw3.h>

void Callbacks::mouse_callback(GLFWwindow *window, double xpos, double ypos) {
    Viewport* viewports = static_cast<Viewport*>(glfwGetWindowUserPointer(window));
    Viewport::viewports[::globalState.selectedViewport]
        .view_camera->calculateDirection(window, xpos, ypos);
}

void Callbacks::scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    Viewport* viewports = static_cast<Viewport*>(glfwGetWindowUserPointer(window));
    Viewport::viewports[::globalState.selectedViewport]
        .view_camera->calculateZoom(yoffset);

}

void Callbacks::key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
        ::globalState.in_view_mode = !(::globalState.in_view_mode);

    //TODO: recheck if necessary or if best method
    if (::globalState.in_view_mode)
        Tools::activeTool = Tools::AvailableTools::None;
}

