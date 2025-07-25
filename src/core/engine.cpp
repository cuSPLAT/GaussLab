#include "engine.h"
#include <stdexcept>

#include <interface/callbacks.h>
#include <interface/viewport.h>
#include <tools/tools.h>

bool GaussLabEngine::initWindow() {
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();

    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);

    m_window = glfwCreateWindow(mode->width, mode->height, "GausStudio", nullptr, nullptr);
    if (!m_window) {
        glfwTerminate();
        return false;
    }
    // For file dialogs
    // NFD should be moved to the interface class
    //if(NFD_Init() != NFD_OKAY) {
    //    std::cerr << "Could not initialize NFD for file dialogs" << std::endl;
    //}
    ////args = {0};
    ////NFD_GetNativeWindowFromGLFWWindow(m_window, &args.parentWindow);

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1);

    // so we can access the renderer from the callbacks
    glfwSetWindowUserPointer(m_window, Viewport::viewports);

    //glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    //-------------------- Callbacks -----------------------
    glfwSetCursorPosCallback(m_window, Callbacks::mouse_callback);
    glfwSetScrollCallback(m_window, Callbacks::scroll_callback);
    glfwSetKeyCallback(m_window, Callbacks::key_callback);
    glfwSetMouseButtonCallback(m_window, Tools::dispatchToTool);

    return true;
}

GaussLabEngine::GaussLabEngine() : interface(marchingCubesEngine, renderer) {
    if (!glfwInit())
        throw std::runtime_error("Failed to initialize GLFW");

    if (!initWindow())
        throw std::runtime_error("Window creation failed");

    if (!renderer.initOpenGL(m_window))
        throw std::runtime_error("Failed to initialize OpenGL");
    renderer.generateInitialBuffers();
}

void run() {

}

GaussLabEngine::~GaussLabEngine() {
    glfwDestroyWindow(m_window);
    glfwTerminate();
}
