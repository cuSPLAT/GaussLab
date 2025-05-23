#pragma once

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <string>

#include <nfd.hpp>

#include <core/renderer.h>
#include <data_reader/dicom_reader.h>

class Interface {
    GLFWwindow* window;
    unsigned int width, height;

    // A pointer for now will be refactored later
    Renderer* renderer;
    DicomReader dcmReader;

    nfdopendialogu8args_t args;

    void setupStyle();

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
