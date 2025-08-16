#pragma once

#include "algorithms/marchingcubes.h"
#include <memory>
#include <optional>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <string>

#include <nfd.hpp>

#include <core/renderer.h>
#include <data_reader/dicom_reader.h>

struct GlobalState;

struct ChatMessage {
    std::string text;
    bool isGemini;
};

struct DicomViewerContext {
    int axialSlice = 0, coronalSlice = 0, sagittalSlice = 0;
    GLuint axialTex = 0, coronalTex = 0, sagittalTex = 0;
    std::vector<unsigned char> axialBuf, coronalBuf, sagittalBuf;
    float windowCenter = 0.0f, windowWidth = 0.0f;
    bool dicom_just_loaded = false;
    bool enableWindowing = true;

    std::vector<ChatMessage> chatLog;
};

class Interface {
    struct GUIContext {
        const int primtiveStepCount = 1000;
        // what if we made the algorithm just choose the max n of threads
        const int log2_threads;
        std::vector<int> allowed_threads;

        int selected_index = 0;
        int n_threads = 1;
        float dropout_p = 0.f;
        glm::vec3 centroid {0}; // a temporary

        GUIContext();
    };
    // Passed from main engine
    MarchingCubesEngine& mc_engine;
    Renderer& renderer;
    GlobalState& appState;

    GUIContext guiCtxt;

    // --------------------
    unsigned int width, height;

    nfdopendialogu8args_t args;

    float uiFontSize = 20.0f;
    
    float windowWidth = 2000.0f;
    float windowCenter = 500.0f;
    int huThreshold = 300;
    int faceCameraIndex = 1;
    float gaussianScale = 1.0f;
    std::string dicomDirectoryPath;

    DicomReader dcmReader;
    DicomViewerContext dcmContext;

public:

    Interface(MarchingCubesEngine& mc, Renderer& renderer, GlobalState& appState);
    ~Interface();

    const DicomReader& getDicomReader() const { return dcmReader; }

    void setupStyle();
    void setupGUI(GLFWwindow* window);
    bool setupWindow();

    void drawInterface();

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
