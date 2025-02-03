#include "nfd.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <string>

#include <nfd.hpp>

#include "renderer.h"

class Interface {
    GLFWwindow* window;
    unsigned int width, height;

    // A pointer for now will be refactored later
    Renderer* renderer;

    nfdopendialogu8args_t args;

public:
    Interface();
    ~Interface();

    bool setupWindow();
    bool initOpengl();

    void setupImgui();
    void setupRenderer();
    void startMainLoop();

    std::string openFileDialog();

    void createMenuBar();
    void createViewWindow();
    void createDockSpace();

};
