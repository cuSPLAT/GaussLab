#define GLFW_INCLUDE_NONE

#include "callbacks.h"

#include <core/renderer.h>

#include <GLFW/glfw3.h>

void Callbacks::mouse_callback(GLFWwindow *window, double xpos, double ypos) {
    Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    renderer->getCamera()->calculateDirection(window, xpos, ypos);
}

void Callbacks::scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    renderer->getCamera()->calculateZoom(yoffset);

}

void Callbacks::key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
        ::globalState.in_view_mode = !(::globalState.in_view_mode);
}

