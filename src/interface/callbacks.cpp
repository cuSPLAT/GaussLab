#include "interface/viewport.h"
#define GLFW_INCLUDE_NONE

#include "callbacks.h"

#include <core/renderer.h>

#include <GLFW/glfw3.h>

void Callbacks::mouse_callback(GLFWwindow *window, double xpos, double ypos) {
    Viewport* viewports = static_cast<Viewport*>(glfwGetWindowUserPointer(window));

    // TODO: Instead of doing a for loop on all of them, just save the
    // id of the hovered viewport and manage it alone
    for (int i = 0; i < Viewport::n_viewports; i++)
        Viewport::viewports[i].view_camera->calculateDirection(window, xpos, ypos);
}

void Callbacks::scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    renderer->getCamera()->calculateZoom(yoffset);

}

void Callbacks::key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
        ::globalState.in_view_mode = !(::globalState.in_view_mode);
}

