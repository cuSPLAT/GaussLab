#pragma once

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <string>

#include <nfd.hpp>

#include <core/renderer.h>
#include <data_reader/dicom_reader.h>
#include "dicom_viewer.h"

struct ChatMessage {
    std::string text;
    bool isGemini;
};

class Interface {
    GLFWwindow* window;
    unsigned int width, height;

    // A pointer for now will be refactored later
    Renderer* renderer;
protected:
    DicomReader dcmReader;

public:
    const DicomReader& getDicomReader() const { return dcmReader; }
    DicomReader& getDicomReader() { return dcmReader; }

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
    void ShowViewerWindow(
        int& axialSlice, int& coronalSlice, int& sagittalSlice,
        GLuint& axialTex, GLuint& coronalTex, GLuint& sagittalTex,
        std::vector<unsigned char>& axialBuf,
        std::vector<unsigned char>& coronalBuf,
        std::vector<unsigned char>& sagittalBuf
    );
    void ShowChatWindow(int axialSlice, std::vector<ChatMessage>& chatLog);
    void ShowDicomViewer();


    std::string openFileDialog();

    void createMenuBar();
    void createViewWindow();
    void createDockSpace();

};
