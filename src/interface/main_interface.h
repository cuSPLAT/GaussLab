#pragma once

#include <memory>
#include <optional>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <string>

#include <nfd.hpp>

#include <core/renderer.h>
#include <data_reader/dicom_reader.h>

struct ChatMessage {
    std::string text;
    bool isGemini;
};

class Interface {
    GLFWwindow* window;
    unsigned int width, height;

    // A pointer for now will be refactored later
    Renderer* renderer;

    nfdopendialogu8args_t args;

    float uiFontSize = 20.0f;
    
    float windowWidth = 2000.0f;
    float windowCenter = 500.0f;
    int huThreshold = 300;
    int faceCameraIndex = 1;
    float gaussianScale = 1.0f;
    std::string dicomDirectoryPath;

    DicomReader dcmReader;

public:

    Interface();
    ~Interface();

    const DicomReader& getDicomReader() const { return dcmReader; }

    void setupStyle();
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
        std::vector<unsigned char>& sagittalBuf,
        float& windowCenter, float& windowWidth,
        bool& enableWindowing
    );
    void ShowChatWindow(int axialSlice, std::vector<ChatMessage>& chatLog);
    void ShowDicomViewer();

    std::optional<std::string> openFileDialog();

    void createMenuBar();
    void createViewWindow();
    void createDockSpace();

};
