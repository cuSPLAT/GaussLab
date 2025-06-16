#include "main_interface.h"
#include <algorithms/marchingcubes.h>
#include "core/renderer.h"
#include "nfd.h"
#include <core/scene_loader.h>
#include <interface/callbacks.h>
#include <debug_utils.h>
#include <tools/tools.h>
#include <chrono>
#include <glm/fwd.hpp>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <nfd.hpp>
#include <nfd_glfw3.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui.h>
#include "ImGuiFileDialog.h"
#include "request.hpp"
#include "dicom_viewer.h"

// DCMTK includes
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcstack.h>

Interface::Interface() = default;

bool Interface::setupWindow() {
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();

    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);

    window = glfwCreateWindow(mode->width, mode->height, "GausStudio", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return false;
    }
    width = mode->width;
    height = mode->height;

    // For file dialogs
    if(NFD_Init() != NFD_OKAY) {
        std::cerr << "Could not initialize NFD for file dialogs" << std::endl;
    }
    args = {0};
    NFD_GetNativeWindowFromGLFWWindow(window, &args.parentWindow);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    renderer = new Renderer(width, height);
    // so we can access the renderer from the callbacks
    glfwSetWindowUserPointer(window, renderer);

    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    //-------------------- Callbacks -----------------------
    glfwSetCursorPosCallback(window, Callbacks::mouse_callback);
    glfwSetScrollCallback(window, Callbacks::scroll_callback);
    glfwSetKeyCallback(window, Callbacks::key_callback);
    glfwSetMouseButtonCallback(window, Tools::dispatchToTool);

    return true;
}

void Interface::setupStyle() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.FrameRounding = 5.f;
    style.WindowRounding = 5.f;
}

bool Interface::initOpengl() {
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return false;
    }

    return true;
}

void Interface::setupImgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");

    this->setupStyle();
    ImGui::StyleColorsDark();
}

std::string Interface::openFileDialog() {
    nfdu8char_t* plyPath;
    
    nfdu8filteritem_t filters[1] = { { "PLY File", "ply" } };
    args.filterList = filters;
    args.filterCount = 1;

    nfdresult_t result = NFD_OpenDialogU8_With(&plyPath, &args);

    std::string path(plyPath);
    NFD_FreePathU8(plyPath);

    return path;
}

void Interface::setupRenderer() {
    renderer->initializeRendererBuffer();
    renderer->generateInitialBuffers();
}

void Interface::createViewWindow() {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
    if (ImGui::Begin("View")) {
        if(ImGui::BeginChild("Render")) {
            if (ImGui::IsWindowHovered()) ::globalState.windowHovered = true;
            else globalState.windowHovered = false;

            ImVec2 viewSize = ImGui::GetWindowSize();
            //TODO: don't always update, just update when window size changes
            renderer->getCamera()->updateViewport(viewSize.x, viewSize.y, renderer->shaderProgram);
            ImGui::Image((ImTextureID)renderer->getRenderBuffer(), viewSize);
        }
        ImGui::EndChild();
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
}

Interface::~Interface() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    NFD_Quit();

    glfwDestroyWindow(window);
    glfwTerminate();

    delete renderer;
}

void Interface::startMainLoop() {
    static int primtiveStepCount = 1000;
    static const int allowed_threads[] = {1, 2, 4};
    static int selected_index = 0;
    static int n_threads = 1;
    static glm::vec3 centroid = glm::vec3(0); // a temporary

    static int axialSlice = 0, coronalSlice = 0, sagittalSlice = 0;
    static GLuint axialTex = 0, coronalTex = 0, sagittalTex = 0;
    static std::vector<unsigned char> axialBuf, coronalBuf, sagittalBuf;
    std::vector<ChatMessage> chatLog;


    /////////////////////////////TO TEST ONLY //////////////////////////////////////
    std::string path = "/home/zain/Downloads/fef7e115-c8c1c523-1e67bffb-13f9223d-1fec008f/55f205a41f404a069013d1728567bc8f Anonymized460/Unknown Study/CT/CT000000.dcm";

    // Load DICOM once
    DcmFileFormat fileFormat;
    OFCondition status = fileFormat.loadFile(path.c_str());
    if (!status.good()) {
        std::cerr << "Failed to load DICOM: " << status.text() << std::endl;
    }

    DcmDataset* dataset = fileFormat.getDataset();
    std::vector<DicomEntry> entries = loadDicomTags(path);
    /////////////////////////////////////////////////////////////////////////////////
    
    while (!glfwWindowShouldClose(window)) {
        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        createMenuBar();
        createDockSpace();
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::SetNextWindowDockID(dockspace_id, ImGuiCond_FirstUseEver);

        createViewWindow();
        ShowViewerWindow(
        axialSlice,coronalSlice, sagittalSlice,
        axialTex, coronalTex, sagittalTex,
        axialBuf,
        coronalBuf,
        sagittalBuf);
        ShowChatWindow(axialSlice, chatLog);
        ShowDicomViewer(entries, dataset);

        if (ImGui::Begin("Debug")) {
            ImGui::InputScalar(
                "Primitive Count", ImGuiDataType_U32, &renderer->verticesCount, &primtiveStepCount
            );
        }
        ImGui::End();

        if(ImGui::Begin("Tabs")) {
            if (ImGui::BeginTabBar("Main Tabs")) {
                // Camera Point View Tab
                if (ImGui::BeginTabItem("Camera Point View")) {
                    static float* viewMat = renderer->getCamera()->getVectorPtr();
                    renderer->getCamera()->getPositionFromShader(renderer->shaderProgram);
                    float position[3] = {viewMat[12], viewMat[13], viewMat[14]};

                    ImGui::Text("Camera Position:");
                    ImGui::InputFloat3("Position", position);

                    // Point cloud or Gaussian splatting view mode selection
                    bool &scene_mode = renderer->getCamera()->scene;
                    if (ImGui::RadioButton("Scene", scene_mode))
                        scene_mode = true;
                    ImGui::SameLine();
                    if (ImGui::RadioButton("Object", !scene_mode))
                        scene_mode = false;
                    // ---------------------------------------------------
                    // Rendering mode selection
                    if (ImGui::BeginCombo("Render Mode", "select")) {
                        if (ImGui::Selectable("PCD"))
                            globalState.debugMode = GL_POINTS;
                            //globalState.renderingMode = GlobalState::RenderMode::PCD;
                        if (ImGui::Selectable("Splats"))
                            //globalState.renderingMode = GlobalState::RenderMode::Splats;
                            globalState.debugMode = GL_TRIANGLES;
                    
                        ImGui::EndCombo();
                    }
                    // ---------------------------------------------------
                    ImGui::Checkbox("Sorting", &globalState.sortingEnabled);
                    ImGui::EndTabItem();
                }
            }
            ImGui::EndTabBar();
            
        }
        ImGui::End();

        // --------------------- Marching Cubes -----------------
        static bool rendered = false;
        if (ImGui::Begin("Marching Cubes")) {
            if (ImGui::SliderInt("Threads", &selected_index, 0, 2, ""))
                n_threads = allowed_threads[selected_index];
            ImGui::SameLine();
            ImGui::Text("%d", allowed_threads[selected_index]);
            if(ImGui::Button("March")) {
                if (dcmReader.loadedData.readable.test_and_set()) {
                    rendered = false;
                    DicomReader::DicomData& data = dcmReader.loadedData;

                    MarchingCubes::marched.clear();
                    MarchingCubes::launchThreaded(
                        data.buffer.get(),
                        data.width, data.length, data.height,
                        660, centroid,
                        1, n_threads
                    );
                }
            }

            static int mc_duration = 0;
            if (MarchingCubes::marched.test_and_set() && !rendered) {
                rendered = true;
                Scene scene;
                scene.centroid = centroid;

                // for debugging
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duration = now - MarchingCubes::last_iter_timer;
                mc_duration = duration.count();

                renderer->constructScene(&scene, MarchingCubes::OutputVertices);
                for (int i = 0; i < MarchingCubes::num_threads; i++)
                    MarchingCubes::TemporaryBuffers[i].clear();
            }
            ImGui::Text("Last run: %d ms", mc_duration);
        }
        // ------------------------------------------------------
        ImGui::End();

        
        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        renderer->render(window);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    dcmReader.cleanupThreads();
    MarchingCubes::cleanUp();
}

void Interface::createMenuBar() {
    // ok maybe this should me moved I don't know
    static bool launchPopup = false;

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open PLY file")) {
                //TODO: error checking
                std::string path = openFileDialog();

                Scene* pcd = PLYLoader::loadPLy(path);
                //renderer->constructScene(pcd);
                delete pcd;
            }
            if (ImGui::MenuItem("Load DICOM Directory")) {
                // ------------------ Folder select ----------------
                nfdu8char_t* path;
                nfdresult_t result = NFD_PickFolderU8(&path, nullptr);

                std::string dir_path(path);
                NFD_FreePathU8(path);
                dcmReader.launchReaderThread(dir_path);
                launchPopup = true;
                // ------------------------------------------------
            }
            if (ImGui::MenuItem("Export OBJ")) {
                if (MarchingCubes::marched.test_and_set()) {
                    //TODO: choose export path
                    DebugUtils::exportObj("output.obj", MarchingCubes::OutputVertices);
                }
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    if (launchPopup)
        ImGui::OpenPopup("dicom_loading");

    if (ImGui::BeginPopupModal("dicom_loading")) {
        ImGui::Text("Loading Dicoms");
        ImGui::ProgressBar((float)dcmReader.loadingProgress/dcmReader.totalSize);

        if (dcmReader.loadingProgress == dcmReader.totalSize && launchPopup) {
            launchPopup = false;
            dcmReader.loadingProgress = 0;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void Interface::createDockSpace() {
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);

    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));

    if (ImGui::Begin("DockSpace Window", nullptr, window_flags)) {
        ImGui::PopStyleVar(4);

        // Define the dock space
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
    }
    ImGui::End();
}

// Texture handles and buffers
// --- Utility: Create OpenGL texture from 2D slice ---
GLuint CreateTextureFromSlice(const unsigned char* data, int width, int height) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Swizzle red to all channels for grayscale display
    GLint swizzleMask[] = {GL_RED, GL_RED, GL_RED, GL_ONE};
    glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);

    return tex;
}

// --- Update textures only when needed ---
void UpdateTextures(const DicomReader::DicomData& dicom,
                    int axialSlice, int coronalSlice, int sagittalSlice,
                    GLuint& axialTex, GLuint& coronalTex, GLuint& sagittalTex,
                    std::vector<unsigned char>& axialBuf,
                    std::vector<unsigned char>& coronalBuf,
                    std::vector<unsigned char>& sagittalBuf) {
    const int width = dicom.width;
    const int height = dicom.length;
    const int depth = dicom.height;

    const float* buffer = dicom.buffer.get();

    // --- Axial (Z) ---
    axialBuf.resize(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t idx = axialSlice * height * width + y * width + x;
            axialBuf[y * width + x] = static_cast<unsigned char>(buffer[idx]);
        }
    }
    axialTex = CreateTextureFromSlice(axialBuf.data(), width, height);

    // --- Coronal (Y) ---
    coronalBuf.resize(width * depth);
    for (int z = 0; z < depth; ++z) {
        for (int x = 0; x < width; ++x) {
            size_t idx = z * height * width + coronalSlice * width + x;
            coronalBuf[z * width + x] = static_cast<unsigned char>(buffer[idx]);
        }
    }
    coronalTex = CreateTextureFromSlice(coronalBuf.data(), width, depth);

    // --- Sagittal (X) ---
    sagittalBuf.resize(height * depth);
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            size_t idx = z * height * width + y * width + sagittalSlice;
            sagittalBuf[z * height + y] = static_cast<unsigned char>(buffer[idx]);
        }
    }
    sagittalTex = CreateTextureFromSlice(sagittalBuf.data(), height, depth);
}

void Interface::ShowViewerWindow(
    int& axialSlice, int& coronalSlice, int& sagittalSlice,
    GLuint& axialTex, GLuint& coronalTex, GLuint& sagittalTex,
    std::vector<unsigned char>& axialBuf,
    std::vector<unsigned char>& coronalBuf,
    std::vector<unsigned char>& sagittalBuf
){
    const auto& dicom = getDicomReader().loadedData;
    const int width = dicom.width;
    const int height = dicom.length;
    const int depth = dicom.height;

    // TODO: Replace these with actual values from your DICOM metadata
    float pixelSpacingX = 1.0f; // mm
    float pixelSpacingY = 1.0f; // mm
    float pixelSpacingZ = 12.0f; // mm

    ImGui::Begin("CT Viewer");

    bool changed = false;

    // Sliders (always reflect current value)
    changed |= ImGui::SliderInt("Axial (Z)", &axialSlice, 0, depth - 1);
    changed |= ImGui::SliderInt("Coronal (Y)", &coronalSlice, 0, height - 1);
    changed |= ImGui::SliderInt("Sagittal (X)", &sagittalSlice, 0, width - 1);

    float viewerWidth = ImGui::GetContentRegionAvail().x;
    float maxSliceWidth = std::min(viewerWidth - 20.0f, 512.0f);

    // Calculate display sizes using pixel spacing
    float axialWidth = maxSliceWidth;
    float axialHeight = maxSliceWidth * (height * pixelSpacingY) / (width * pixelSpacingX);

    float coronalWidth = maxSliceWidth;
    float coronalHeight = maxSliceWidth * (depth * pixelSpacingZ) / (width * pixelSpacingX);

    float sagittalWidth = maxSliceWidth;
    float sagittalHeight = maxSliceWidth * (depth * pixelSpacingZ) / (height * pixelSpacingY);

    // Axial View
    ImGui::PushID("axial");
    ImGui::Text("Axial");
    ImVec2 axialPos = ImGui::GetCursorScreenPos();
    ImGui::Image((ImTextureID)(intptr_t)axialTex, ImVec2(axialWidth, axialHeight));
    ImVec2 axialEnd = ImGui::GetCursorScreenPos();
    if (ImGui::IsMouseHoveringRect(axialPos, axialEnd)) {
        float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f) {
            int old = axialSlice;
            axialSlice = std::clamp(axialSlice - static_cast<int>(wheel), 0, depth - 1);
            if (axialSlice != old) changed = true;
        }
    }
    ImGui::PopID();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Coronal View
    ImGui::PushID("coronal");
    ImGui::Text("Coronal");
    ImVec2 coronalPos = ImGui::GetCursorScreenPos();
    ImGui::Image((ImTextureID)(intptr_t)coronalTex, ImVec2(coronalWidth, coronalHeight));
    ImVec2 coronalEnd = ImGui::GetCursorScreenPos();
    if (ImGui::IsMouseHoveringRect(coronalPos, coronalEnd)) {
        float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f) {
            int old = coronalSlice;
            coronalSlice = std::clamp(coronalSlice - static_cast<int>(wheel), 0, height - 1);
            if (coronalSlice != old) changed = true;
        }
    }
    ImGui::PopID();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Sagittal View
    ImGui::PushID("sagittal");
    ImGui::Text("Sagittal");
    ImVec2 sagittalPos = ImGui::GetCursorScreenPos();
    ImGui::Image((ImTextureID)(intptr_t)sagittalTex, ImVec2(sagittalWidth, sagittalHeight));
    ImVec2 sagittalEnd = ImGui::GetCursorScreenPos();
    if (ImGui::IsMouseHoveringRect(sagittalPos, sagittalEnd)) {
        float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f) {
            int old = sagittalSlice;
            sagittalSlice = std::clamp(sagittalSlice - static_cast<int>(wheel), 0, width - 1);
            if (sagittalSlice != old) changed = true;
        }
    }
    ImGui::PopID();

    // Update textures if any change (slider or scroll)
    if (changed) {
        if (axialTex) glDeleteTextures(1, &axialTex);
        if (coronalTex) glDeleteTextures(1, &coronalTex);
        if (sagittalTex) glDeleteTextures(1, &sagittalTex);

        UpdateTextures(dicom, axialSlice, coronalSlice, sagittalSlice,
                       axialTex, coronalTex, sagittalTex,
                       axialBuf, coronalBuf, sagittalBuf);
    }

    ImGui::End();
}





// ImGui chat interface
// Include ImGuiFileDialog

void Interface::ShowChatWindow(int axialSlice, std::vector<ChatMessage>& chatLog) {
    static char inputBuffer[1024] = "";  // User input for prompt
    static bool shouldScrollToBottom = false;

    ImGui::Begin("Gemini Assistant", nullptr, ImGuiWindowFlags_NoCollapse);

    // Calculate available height for chat log (window height minus input area and padding)
    float windowHeight = ImGui::GetWindowHeight();
    float inputHeight = 40.0f;  // Height for input area
    float padding = 10.0f;      // Padding between elements
    float chatLogHeight = windowHeight - inputHeight - padding * 2;

    // Chat Log Area
    ImGui::BeginChild("ChatLog", ImVec2(0, chatLogHeight), true, ImGuiWindowFlags_HorizontalScrollbar);
    
    // Add some padding at the top
    ImGui::Spacing();
    
    for (const auto& message : chatLog) {
        // Calculate text width for wrapping
        float wrapWidth = ImGui::GetWindowWidth() - 20.0f;  // Leave some margin
        
        // Style for user messages
        if (!message.isGemini) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 0.8f, 1.0f));  // Light gray for user
            ImGui::TextWrapped("%s", message.text.c_str());
            ImGui::PopStyleColor();
        }
        // Style for Gemini messages
        else {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.2f, 0.6f, 1.0f, 1.0f));  // Light blue for Gemini
            ImGui::TextWrapped("%s", message.text.c_str());
            ImGui::PopStyleColor();
        }
        
        // Add spacing between messages
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
    }

    // Auto-scroll to bottom when new messages are added
    if (shouldScrollToBottom) {
        ImGui::SetScrollHereY(1.0f);
        shouldScrollToBottom = false;
    }

    ImGui::EndChild();

    // Input Area
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Create a horizontal layout for input and button
    float buttonWidth = 80.0f;
    float inputWidth = ImGui::GetWindowWidth() - buttonWidth - padding * 2;

    ImGui::PushItemWidth(inputWidth);
    if (ImGui::InputText("##Input", inputBuffer, IM_ARRAYSIZE(inputBuffer), 
                        ImGuiInputTextFlags_EnterReturnsTrue)) {
        // Handle Enter key press
        std::string inputText(inputBuffer);
        if (!inputText.empty()) {
            chatLog.push_back({"You: " + inputText, false});
            const auto& dicom = getDicomReader().loadedData;
            std::string response = getGeminiResponseWithImage(inputText, dicom, axialSlice);
            chatLog.push_back({"Gemini: " + response, true});
            inputBuffer[0] = '\0';
            shouldScrollToBottom = true;
        }
    }
    ImGui::PopItemWidth();

    ImGui::SameLine();
    if (ImGui::Button("Send", ImVec2(buttonWidth, 0))) {
        std::string inputText(inputBuffer);
        if (!inputText.empty()) {
            chatLog.push_back({"You: " + inputText, false});
            const auto& dicom = getDicomReader().loadedData;
            std::string response = getGeminiResponseWithImage(inputText, dicom, axialSlice);
            chatLog.push_back({"Gemini: " + response, true});
            inputBuffer[0] = '\0';
            shouldScrollToBottom = true;
        }
        else {
            chatLog.push_back({"Gemini: Please enter a prompt.", true});
            shouldScrollToBottom = true;
        }
    }

    ImGui::End();
}



