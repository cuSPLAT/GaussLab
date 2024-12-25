#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "renderer.h"

class Interface {
    GLFWwindow* window;
    unsigned int width, height;

    // A pointer for now will be refactored later
    Renderer* renderer;

public:
    Interface();
    ~Interface();

    bool setupWindow();
    bool initOpengl();

    void setupImgui();
    void setupRenderer();
    void startMainLoop();

    void createMenuBar();
    void createViewWindow();
    void createDockSpace();

};
