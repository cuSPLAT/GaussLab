#define GLFW_INCLUDE_NONE

#include "callbacks.h"
#include "renderer.h"

#include <GLFW/glfw3.h>

void Callbacks::mouse_callback(GLFWwindow *window, double xpos, double ypos) {
    Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    renderer->getCamera()->calculateDirection(window, xpos, ypos);
}

void Callbacks::scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    renderer->getCamera()->calculateZoom(yoffset);

}

