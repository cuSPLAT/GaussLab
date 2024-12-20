#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

class Interface {
    GLFWwindow* window;

public:
    Interface() = default;
    ~Interface();

    bool setupWindow();
    bool initOpengl();

    void setupImgui();
    void startMainLoop();

    void createMenuBar();
    void createViewWindow();
    void createDockSpace();

};
