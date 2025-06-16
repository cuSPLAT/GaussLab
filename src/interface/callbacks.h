#pragma once

#include <GLFW/glfw3.h>

namespace Callbacks {
    void mouse_callback(GLFWwindow*  window, double xpos, double ypos);
    void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods);
};
