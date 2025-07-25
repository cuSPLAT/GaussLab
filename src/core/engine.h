#pragma once

#include <GLFW/glfw3.h>
#include <algorithms/marchingcubes.h>
#include <core/renderer.h>
#include <interface/main_interface.h>


// A global state for the renderer, this will be used to pass data between classes
// instead of always having to call a function from one class with certain parameters
// which would get messy after time
struct GlobalState {
    enum class RenderMode {
        Splats,
        PCD
    };

    // write from GUI only
    RenderMode renderingMode = RenderMode::Splats;
    bool sortingEnabled = true;
    bool windowHovered = false;
    bool in_view_mode = true;

    // on start the first viewport is selected
    int selectedViewport = 0;
    int available_threads = std::thread::hardware_concurrency();

    GLuint vertexProgram, gaussianProgram;

    GLuint debugMode = GL_TRIANGLES;
};

class GaussLabEngine {
    Renderer renderer;
    Interface interface;
    MarchingCubesEngine marchingCubesEngine;

    GLFWwindow* m_window;

public:
    GaussLabEngine();
    ~GaussLabEngine();

    void run();

private:
    bool initWindow();
};
