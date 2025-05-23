#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <interface/main_interface.h>

int main() {

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    Interface interface;
    interface.setupWindow();
    interface.initOpengl();
    interface.setupImgui();
    interface.setupRenderer();
    interface.startMainLoop();

    return 0;
}
