#include "main_interface.h"

#include <algorithms/marchingcubes.h>
#include "core/renderer.h"
#include "nfd.h"
#include <cmath>
#include <core/scene_loader.h>
#include <cstdio>
#include <interface/callbacks.h>
#include <interface/viewport.h>

#include <debug_utils.h>

#include <optional>
#include <tools/tools.h>
#include "backend/includes/model.hpp"
#include "backend/includes/dicom_loader.hpp"

#include <chrono>
#include <glm/fwd.hpp>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/gtc/type_ptr.hpp>
#include <nfd.hpp>
#include <nfd_glfw3.h>

#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui.h>

#include "request.hpp"
#include "dicom_viewer.h"

void UpdateTextures(const DicomReader::DicomData& dicom,
                    int axialSlice, int coronalSlice, int sagittalSlice,
                    GLuint& axialTex, GLuint& coronalTex, GLuint& sagittalTex,
                    std::vector<unsigned char>& axialBuf,
                    std::vector<unsigned char>& coronalBuf,
                    std::vector<unsigned char>& sagittalBuf,
                    float windowCenter, float windowWidth,
                    bool enableWindowing);


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
    glfwSetWindowUserPointer(window, Viewport::viewports);

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
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    return true;
}

void Interface::setupImgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    const char* font_path = "../assets/JetBrainsMono-Regular.ttf";
    //float font_size = 18.0f;
    io.Fonts->AddFontFromFileTTF(font_path, uiFontSize);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");

    this->setupStyle();
    ImGui::StyleColorsDark();
}

std::optional<std::string> Interface::openFileDialog() {
    nfdu8char_t* plyPath;
    
    nfdu8filteritem_t filters[1] = { { "PLY File", "ply" } };
    args.filterList = filters;
    args.filterCount = 1;

    nfdresult_t result = NFD_OpenDialogU8_With(&plyPath, &args);
    if (result == NFD_OKAY) {
        std::string path(plyPath);
        NFD_FreePathU8(plyPath);
        return std::move(path);
    }

    return std::nullopt;
}

void Interface::setupRenderer() {
    renderer->generateInitialBuffers();
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
    const int primtiveStepCount = 1000;
    
    // could be done with a range to be faster
    const int log2_threads = std::log2(::globalState.available_threads) + 1;
    std::vector<int> allowed_threads(log2_threads);
    for (int i = 0; i < log2_threads; i++)
        allowed_threads[i] = std::pow(2, i);
    int selected_index = 0;
    int n_threads = 1;

    float dropout_p = 0.f;

    glm::vec3 centroid {0}; // a temporary

    static int axialSlice = 0, coronalSlice = 0, sagittalSlice = 0;
    static GLuint axialTex = 0, coronalTex = 0, sagittalTex = 0;
    static std::vector<unsigned char> axialBuf, coronalBuf, sagittalBuf;
    static float windowCenter = 0.0f, windowWidth = 0.0f;
    static bool dicom_just_loaded = false;
    static bool enableWindowing = true;
    std::vector<ChatMessage> chatLog;

    // The main initial viewport
    Viewport::newViewport(width, height);

    while (!glfwWindowShouldClose(window)) {
        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (dicom_just_loaded) {
            const auto& dicom = dcmReader.loadedData;
            windowCenter = dicom.windowCenter;
            windowWidth = dicom.windowWidth;
            axialSlice = dicom.height / 2;
            coronalSlice = dicom.length / 2;
            sagittalSlice = dicom.width / 2;
            
            if (axialTex) glDeleteTextures(1, &axialTex);
            if (coronalTex) glDeleteTextures(1, &coronalTex);
            if (sagittalTex) glDeleteTextures(1, &sagittalTex);
            axialTex = coronalTex = sagittalTex = 0;

            UpdateTextures(dicom, axialSlice, coronalSlice, sagittalSlice,
                           axialTex, coronalTex, sagittalTex,
                           axialBuf, coronalBuf, sagittalBuf,
                           windowCenter, windowWidth, enableWindowing);
            dicom_just_loaded = false;
        }

        //ImGui::ShowMetricsWindow();

        createMenuBar();
        createDockSpace();
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::SetNextWindowDockID(dockspace_id, ImGuiCond_FirstUseEver);

        // ----------------------------- Drawing ------------------------
        Viewport::drawViewports_ImGui(renderer);
        Viewport& selectedViewport = Viewport::viewports[::globalState.selectedViewport];

        // ------------------------------ DICOM Viewer -----------------
        ShowViewerWindow(
            axialSlice, coronalSlice, sagittalSlice,
            axialTex, coronalTex, sagittalTex,
            axialBuf,
            coronalBuf,
            sagittalBuf,
            windowCenter, windowWidth, enableWindowing);
        ShowChatWindow(axialSlice, chatLog);
        // only reload dicom tag entries if the file changes
        ShowDicomViewer();
        // ---------------------------------------------------------------

        if (ImGui::Begin("Debug")) {
            ImGui::SeparatorText("Primitive Count");
            ImGui::PushItemWidth(-1.0f);
            ImGui::InputScalar(
                "##PrimitiveCount", ImGuiDataType_U32, &renderer->verticesCount, &primtiveStepCount
            );
        }
        ImGui::End();

        if(ImGui::Begin("Tabs")) {
            if (ImGui::BeginTabBar("Main Tabs")) {
                // Camera Point View Tab
                if (ImGui::BeginTabItem("Camera Point View")) {
                    ImGui::Text("Camera Position:");
                    ImGui::PushItemWidth(-1.0f);
                    ImGui::InputFloat3("##Position", glm::value_ptr(selectedViewport.view_camera->cameraPos));

                    if (ImGui::Button("Reset")) {
                        Viewport::lookAtScene_all(centroid);
                    }

                    // Point cloud or Gaussian splatting view mode selection
                    ImGui::Dummy(ImVec2(0.f, 10.f));
                    ImGui::SeparatorText("Viewport");
                    if (ImGui::RadioButton("Scene", selectedViewport.view_camera->scene))
                        selectedViewport.view_camera->scene = true;
                    ImGui::SameLine();
                    if (ImGui::RadioButton("Object", !selectedViewport.view_camera->scene))
                        selectedViewport.view_camera->scene = false;
                    // ---------------------------------------------------
                    // Rendering mode selection
                    ImGui::Dummy(ImVec2(0.f, 10.f));
                    ImGui::Text("Render Mode: ");
                    if (ImGui::BeginCombo("##RenderMode", "select")) {
                        if (ImGui::Selectable("PCD"))
                            //globalState.debugMode = GL_POINTS;
                            globalState.renderingMode = GlobalState::RenderMode::PCD;
                        if (ImGui::Selectable("Splats"))
                            globalState.renderingMode = GlobalState::RenderMode::Splats;
                            //globalState.debugMode = GL_TRIANGLES;
                    
                        ImGui::EndCombo();
                    }
                    // ---------------------------------------------------
                    ImGui::Dummy(ImVec2(0.f, 10.f));
                    ImGui::SeparatorText("Gaussian Splats");
                    ImGui::Checkbox("Splat Sorting", &globalState.sortingEnabled);

                    // -------------------- Toolbox ----------------------
                    ImGui::Dummy(ImVec2(0.f, 10.f));
                    ImGui::SeparatorText("Tool Box");
                    Tools::drawToolBox_ImGui();
                    
                    // -------------------- 3DGS Construction ----------------------
                    ImGui::Dummy(ImVec2(0.f, 10.f));
                    ImGui::SeparatorText("3DGS Construction");

                    float label_width = 120.0f;
                    // --- Window Width ---
                    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - label_width);
                    ImGui::Text("Window Width");
                    ImGui::SameLine(label_width);
                    ImGui::InputFloat("##WW", &windowWidth, 0.0f, 0.0f, "%.1f"); // "##WW" makes the label invisible
                    ImGui::PopItemWidth();

                    // --- Window Center ---
                    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - label_width);
                    ImGui::Text("Window Center");
                    ImGui::SameLine(label_width);
                    ImGui::InputFloat("##WC", &windowCenter, 0.0f, 0.0f, "%.1f");
                    ImGui::PopItemWidth();

                    // --- HU Threshold ---
                    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - label_width);
                    ImGui::Text("HU Threshold");
                    ImGui::SameLine(label_width);
                    ImGui::InputInt("##HU", &huThreshold);
                    ImGui::SliderFloat("Dropout", &dropout_p, 0.f, 1.f);
                    ImGui::PopItemWidth();

                    // --- Face Camera Slider ---
                    const char* camera_options[] = { "+X", "+Y", "+Z" };
                    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - label_width);
                    ImGui::Text("Face Camera");
                    ImGui::SameLine(label_width);
                    ImGui::SliderInt("##FaceCamera", &faceCameraIndex, 0, 2, camera_options[faceCameraIndex]);
                    ImGui::PopItemWidth();

                    ImGui::Dummy(ImVec2(0.f, 5.f));
                    float dcmDirButtonWidth = 80.0f;
                    float spacing = ImGui::GetStyle().ItemSpacing.x;
                    float fullWidth = ImGui::GetContentRegionAvail().x;
                    if (ImGui::Button("dcmDir", ImVec2(dcmDirButtonWidth, 0)))
                    {
                        nfdu8char_t* path;
                        nfdresult_t result = NFD_PickFolderU8(&path, nullptr);
                    
                        if (result == NFD_OKAY)
                        {
                            dicomDirectoryPath = std::string(path); 
                            NFD_FreePathU8(path); // Free the path returned by NFD
                            printf("Selected DICOM Directory: %s\n", dicomDirectoryPath.c_str());
                        }
                        else if (result == NFD_CANCEL)
                        {
                            printf("CANCELLED.\n");
                        } 
                        else
                        {
                            printf("Error picking directory: %s\n", NFD_GetError());
                        }
                    }

                    ImGui::SameLine();
                    if (ImGui::Button("Generate Splats", ImVec2(fullWidth - dcmDirButtonWidth - spacing, 0)))
                    { // -1 width makes it span the full width
                        if (dicomDirectoryPath.empty())
                        {
                            printf("Error: Please select a DICOM directory first.\n");
                        }
                        else
                        {
                            torch::Device device = torch::kCUDA;
                            std::cout << device << "\n";
                            InputData inputData = inputDataFromDicom(
                                dicomDirectoryPath, windowWidth, windowCenter,
                                huThreshold, 1, faceCameraIndex, dropout_p
                            );
                            Model model(inputData, device);
                            GPUScene reconstructedScene = {
                                .means = model.means,
                                .colors = model.featuresDc,
                                .opacities = 1 / (1 + torch::exp(-model.opacities)),
                                .scales = torch::exp(model.scales),
                                .centroid = {model.centroid_f[0], model.centroid_f[1], model.centroid_f[2]}
                            };
                            renderer->constructSplatSceneFromGPU(reconstructedScene);
                        }
                    }

                    ImGui::Dummy(ImVec2(0.f, 10.f));
                    ImGui::SeparatorText("Gaussian Scailing");
                    ImGui::SliderFloat("##GaussianScale", &gaussianScale, 0.0001f, 1.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
                    ImGui::PopItemWidth();

                    ImGui::EndTabItem();
                }
            }
            ImGui::EndTabBar();
            
        }
        ImGui::End();

        // --------------------- Marching Cubes -----------------
        static bool rendered = false;
        static int HU_threshold = 660;
        if (ImGui::Begin("Marching Cubes")) {
            if (ImGui::SliderInt("Threads", &selected_index, 0, log2_threads - 1, ""))
                n_threads = allowed_threads[selected_index];
            ImGui::SameLine();
            ImGui::Text("%d", allowed_threads[selected_index]);
            ImGui::InputScalar("HU value", ImGuiDataType_S32, &HU_threshold);
            if(ImGui::Button("March")) {
                if (dcmReader.loadedData.readable.test()) {
                    rendered = false;
                    const DicomReader::DicomData& data = dcmReader.loadedData;

                    MarchingCubes::marched.clear();
                    MarchingCubes::launchThreaded(
                        data.buffer.get(),
                        data.width, data.length, data.height,
                        HU_threshold, centroid,
                        1, n_threads
                    );
                }
            }

            static int mc_duration = 0;
            if (MarchingCubes::marched.test() && !rendered) {
                rendered = true;
                // for debugging
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duration = now - MarchingCubes::last_iter_timer;
                mc_duration = duration.count();

                renderer->constructMeshScene(MarchingCubes::OutputVertices);

                for (int i = 0; i < Viewport::n_viewports; i++) {
                    if (Viewport::viewports[i].mesh)
                        Viewport::viewports[i].view_camera->lookAt(centroid);
                }
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
                std::optional<std::string> path = openFileDialog();
                if (path.has_value()) {
                    Scene* pcd = PLYLoader::loadPLy(std::move(path.value()));
                    renderer->constructSplatScene(pcd);
                    delete pcd;
                }
            }
            if (ImGui::MenuItem("Load DICOM Directory")) {
                // ------------------ Folder select ----------------
                nfdu8char_t* path;
                nfdresult_t result = NFD_PickFolderU8(&path, nullptr);

                if (result == NFD_OKAY) {
                    std::string dir_path(path);
                    NFD_FreePathU8(path);
                    dcmReader.launchReaderThread(std::move(dir_path));
                    launchPopup = true;
                }
                // ------------------------------------------------
            }
            if (ImGui::BeginMenu("Export")) {
                if (ImGui::MenuItem("Wavefront (.obj)")) {
                    if (MarchingCubes::marched.test()) {
                        //TODO: choose export path
                        DebugUtils::exportObj("output.obj", MarchingCubes::OutputVertices);
                    }
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("New Viewport")) {
                // A maximum of 5 windows is allowed
                if (Viewport::n_viewports != MAX_VIEWPORTS)
                    Viewport::newViewport(width, height);
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
                    std::vector<unsigned char>& sagittalBuf,
                    float windowCenter, float windowWidth,
                    bool enableWindowing) {
    const int width = dicom.width;
    const int height = dicom.length;
    const int depth = dicom.height;

    const float* buffer = dicom.buffer.get();

    if (enableWindowing) {
        float lower = windowCenter - windowWidth / 2.0f;
        float upper = windowCenter + windowWidth / 2.0f;
        float range = upper - lower;
        if (range <= 1e-5) range = 1.0f;

        // --- Axial (Z) ---
        axialBuf.resize(width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                size_t idx = axialSlice * height * width + y * width + x;
                float val = buffer[idx];
                float norm = (val - lower) / range;
                axialBuf[y * width + x] = static_cast<unsigned char>(std::clamp(norm, 0.0f, 1.0f) * 255);
            }
        }

        // --- Coronal (Y) ---
        coronalBuf.resize(width * depth);
        for (int z = 0; z < depth; ++z) {
            for (int x = 0; x < width; ++x) {
                size_t idx = z * height * width + coronalSlice * width + x;
                float val = buffer[idx];
                float norm = (val - lower) / range;
                coronalBuf[z * width + x] = static_cast<unsigned char>(std::clamp(norm, 0.0f, 1.0f) * 255);
            }
        }

        // --- Sagittal (X) ---
        sagittalBuf.resize(height * depth);
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                size_t idx = z * height * width + y * width + sagittalSlice;
                float val = buffer[idx];
                float norm = (val - lower) / range;
                sagittalBuf[z * height + y] = static_cast<unsigned char>(std::clamp(norm, 0.0f, 1.0f) * 255);
            }
        }
    } else {
        float range = dicom.dataMax - dicom.dataMin;
        if (range <= 1e-5) range = 1.0f;
        // --- Axial (Z) ---
        axialBuf.resize(width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                size_t idx = axialSlice * height * width + y * width + x;
                float norm = (buffer[idx] - dicom.dataMin) / range;
                axialBuf[y * width + x] = static_cast<unsigned char>(std::clamp(norm * 255.0f, 0.0f, 255.0f));
            }
        }

        // --- Coronal (Y) ---
        coronalBuf.resize(width * depth);
        for (int z = 0; z < depth; ++z) {
            for (int x = 0; x < width; ++x) {
                size_t idx = z * height * width + coronalSlice * width + x;
                float norm = (buffer[idx] - dicom.dataMin) / range;
                coronalBuf[z * width + x] = static_cast<unsigned char>(std::clamp(norm * 255.0f, 0.0f, 255.0f));
            }
        }

        // --- Sagittal (X) ---
        sagittalBuf.resize(height * depth);
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                size_t idx = z * height * width + y * width + sagittalSlice;
                float norm = (buffer[idx] - dicom.dataMin) / range;
                sagittalBuf[z * height + y] = static_cast<unsigned char>(std::clamp(norm * 255.0f, 0.0f, 255.0f));
            }
        }
    }

    axialTex = CreateTextureFromSlice(axialBuf.data(), width, height);
    coronalTex = CreateTextureFromSlice(coronalBuf.data(), width, depth);
    sagittalTex = CreateTextureFromSlice(sagittalBuf.data(), height, depth);
}

void Interface::ShowViewerWindow(
    int& axialSlice, int& coronalSlice, int& sagittalSlice,
    GLuint& axialTex, GLuint& coronalTex, GLuint& sagittalTex,
    std::vector<unsigned char>& axialBuf,
    std::vector<unsigned char>& coronalBuf,
    std::vector<unsigned char>& sagittalBuf,
    float& windowCenter, float& windowWidth,
    bool& enableWindowing
){
    const auto& dicom = getDicomReader().loadedData;
    
    // Check if DICOM data is valid before accessing it
    if (!dicom.buffer || dicom.width <= 0 || dicom.length <= 0 || dicom.height <= 0) {
        ImGui::Begin("CT Viewer");
        ImGui::Text("Please load a DICOM directory first");
        ImGui::End();
        return;
    }
    
    const int width = dicom.width;
    const int height = dicom.length;
    const int depth = dicom.height;

    // Clamp slice indices to valid ranges
    axialSlice = std::clamp(axialSlice, 0, depth - 1);
    coronalSlice = std::clamp(coronalSlice, 0, height - 1);
    sagittalSlice = std::clamp(sagittalSlice, 0, width - 1);

    float pixelSpacingX = dicom.pixelSpacingX;
    float pixelSpacingY = dicom.pixelSpacingY;
    float pixelSpacingZ = dicom.sliceThickness;

    ImGui::Begin("CT Viewer");

    bool changed = false;

    // Windowing controls
    changed |= ImGui::Checkbox("Enable Windowing", &enableWindowing);
    if (enableWindowing) {
        changed |= ImGui::SliderFloat("Window Center", &windowCenter, dicom.dataMin, dicom.dataMax);
        changed |= ImGui::SliderFloat("Window Width", &windowWidth, 0, dicom.dataMax - dicom.dataMin);
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Axial View
    ImGui::PushID("axial");
    ImGui::Text("Axial");
    changed |= ImGui::SliderInt("Axial (Z)", &axialSlice, 0, depth - 1);
    float axialWidth = std::min(ImGui::GetContentRegionAvail().x, 512.0f);
    float axialHeight = axialWidth * (height * pixelSpacingY) / (width * pixelSpacingX);
    ImGui::Image((ImTextureID)(intptr_t)axialTex, ImVec2(axialWidth, axialHeight));
    ImGui::PopID();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Coronal View
    ImGui::PushID("coronal");
    ImGui::Text("Coronal");
    changed |= ImGui::SliderInt("Coronal (Y)", &coronalSlice, 0, height - 1);
    float coronalWidth = std::min(ImGui::GetContentRegionAvail().x, 512.0f);
    float coronalHeight = coronalWidth * (depth * pixelSpacingZ) / (width * pixelSpacingX);
    ImGui::Image((ImTextureID)(intptr_t)coronalTex, ImVec2(coronalWidth, coronalHeight));
    ImGui::PopID();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Sagittal View
    ImGui::PushID("sagittal");
    ImGui::Text("Sagittal");
    changed |= ImGui::SliderInt("Sagittal (X)", &sagittalSlice, 0, width - 1);
    float sagittalWidth = std::min(ImGui::GetContentRegionAvail().x, 512.0f);
    float sagittalHeight = sagittalWidth * (depth * pixelSpacingZ) / (height * pixelSpacingY);
    ImGui::Image((ImTextureID)(intptr_t)sagittalTex, ImVec2(sagittalWidth, sagittalHeight));
    ImGui::PopID();

    // Update textures if any change (slider or scroll)
    if (changed) {
        if (axialTex) glDeleteTextures(1, &axialTex);
        if (coronalTex) glDeleteTextures(1, &coronalTex);
        if (sagittalTex) glDeleteTextures(1, &sagittalTex);

        UpdateTextures(dicom, axialSlice, coronalSlice, sagittalSlice,
                       axialTex, coronalTex, sagittalTex,
                       axialBuf, coronalBuf, sagittalBuf,
                       windowCenter, windowWidth, enableWindowing);
    }

    ImGui::End();
}





// ImGui chat interface
// Include ImGuiFileDialog

void Interface::ShowChatWindow(int axialSlice, std::vector<ChatMessage>& chatLog) {
    static char inputBuffer[1024] = "";  // User input for prompt
    static bool shouldScrollToBottom = false;
    static int typingPosition = 0;
    static float typingTimer = 0.0f;
    static std::string currentTypingMessage = "";

    ImGui::Begin("GaussLab Assistant", nullptr, ImGuiWindowFlags_NoCollapse);

    float inputHeight = ImGui::GetFrameHeightWithSpacing() * 2; // or a small constant if you want
    float chatLogHeight = ImGui::GetContentRegionAvail().y - inputHeight;
    ImGui::BeginChild("ChatLog", ImVec2(0, chatLogHeight), true, ImGuiWindowFlags_HorizontalScrollbar);
    
    // Add some padding at the top
    ImGui::Spacing();
    
    for (size_t i = 0; i < chatLog.size(); ++i) {
        const auto& message = chatLog[i];
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
            if (i == chatLog.size() - 1 && !currentTypingMessage.empty()) {
                std::string displayText = "GaussLab: " + currentTypingMessage.substr(0, typingPosition);
                ImGui::TextWrapped("%s", displayText.c_str());
            } else {
                ImGui::TextWrapped("%s", message.text.c_str());
            }
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
    float padding = 10.0f;
    float inputWidth = ImGui::GetWindowWidth() - buttonWidth - padding * 2;

    ImGui::PushItemWidth(inputWidth);
    if (ImGui::InputText("##Input", inputBuffer, IM_ARRAYSIZE(inputBuffer), 
                        ImGuiInputTextFlags_EnterReturnsTrue)) {
        // Handle Enter key press
        std::string inputText(inputBuffer);
        if (!inputText.empty()) {
            // Display user message immediately
            chatLog.push_back({"You: " + inputText, false});
            shouldScrollToBottom = true;
            inputBuffer[0] = '\0';
            
            // Get response asynchronously
            const auto& dicom = getDicomReader().loadedData;
            
            std::stringstream prompt;
            prompt << "You are a radiology assistant. Analyze the following DICOM slice with the given metadata and answer the user's question.\n\n";
            prompt << "--- DICOM Metadata ---\n";
            if (!dicom.patientName.empty()) prompt << "Patient Name: " << dicom.patientName << "\n";
            if (!dicom.scanDate.empty()) prompt << "Date of Scan: " << dicom.scanDate << "\n";
            prompt << "Type of Scan: Axial CT Scan\n";
            if (!dicom.bodyPartExamined.empty()) prompt << "Region Scanned: " << dicom.bodyPartExamined << "\n";
            if (!dicom.contrastAgent.empty()) prompt << "Type of CT scan: Contrast-enhanced\n";
            else prompt << "Type of CT scan: Non-contrast\n";
            if (dicom.sliceThickness > 0) prompt << "Slice thickness: " << dicom.sliceThickness << " mm axial sections\n";
            if (!dicom.reasonForStudy.empty()) prompt << "Reason for Study: " << dicom.reasonForStudy << "\n";
            if (!dicom.reasonForRequestedProcedure.empty()) prompt << "Reason for the Requested Procedure: " << dicom.reasonForRequestedProcedure << "\n";
            if (!dicom.requestedProcedureDescription.empty()) prompt << "Requested Procedure Description: " << dicom.requestedProcedureDescription << "\n";
            prompt << "\n--- User Question ---\n";
            prompt << inputText;

            std::string response = getGeminiResponseWithImage(prompt.str(), dicom, axialSlice);
            // Start typing effect for the response
            currentTypingMessage = response;
            typingPosition = 0;
            typingTimer = 0.0f;
            chatLog.push_back({"Gemini: " + response, true});
        }
    }
    ImGui::PopItemWidth();

    ImGui::SameLine();
    if (ImGui::Button("Send", ImVec2(buttonWidth, 0))) {
        std::string inputText(inputBuffer);
        if (!inputText.empty()) {
            // Display user message immediately
            chatLog.push_back({"You: " + inputText, false});
            shouldScrollToBottom = true;
            inputBuffer[0] = '\0';
            
            // Get response asynchronously
            const auto& dicom = getDicomReader().loadedData;
            
            std::stringstream prompt;
            prompt << "You are a radiology assistant. Analyze the following DICOM slice with the given metadata and answer the user's question.\n\n";
            prompt << "--- DICOM Metadata ---\n";
            if (!dicom.patientName.empty()) prompt << "Patient Name: " << dicom.patientName << "\n";
            if (!dicom.scanDate.empty()) prompt << "Date of Scan: " << dicom.scanDate << "\n";
            prompt << "Type of Scan: Axial CT Scan\n";
            if (!dicom.bodyPartExamined.empty()) prompt << "Region Scanned: " << dicom.bodyPartExamined << "\n";
            if (!dicom.contrastAgent.empty()) prompt << "Type of CT scan: Contrast-enhanced\n";
            else prompt << "Type of CT scan: Non-contrast\n";
            if (dicom.sliceThickness > 0) prompt << "Slice thickness: " << dicom.sliceThickness << " mm axial sections\n";
            if (!dicom.reasonForStudy.empty()) prompt << "Reason for Study: " << dicom.reasonForStudy << "\n";
            if (!dicom.reasonForRequestedProcedure.empty()) prompt << "Reason for the Requested Procedure: " << dicom.reasonForRequestedProcedure << "\n";
            if (!dicom.requestedProcedureDescription.empty()) prompt << "Requested Procedure Description: " << dicom.requestedProcedureDescription << "\n";
            prompt << "\n--- User Question ---\n";
            prompt << inputText;

            std::string response = getGeminiResponseWithImage(prompt.str(), dicom, axialSlice);
            // Start typing effect for the response
            currentTypingMessage = response;
            typingPosition = 0;
            typingTimer = 0.0f;
            chatLog.push_back({"Gemini: " + response, true});
        }
        else {
            chatLog.push_back({"Gemini: Please enter a prompt.", true});
            shouldScrollToBottom = true;
        }
    }

    // Update typing effect
    if (!currentTypingMessage.empty() && typingPosition < currentTypingMessage.length()) {
        typingTimer += ImGui::GetIO().DeltaTime;
        if (typingTimer >= 0.02f) { // Faster typing (0.02 = 50 characters per second)
            typingPosition++;
            typingTimer = 0.0f;
            shouldScrollToBottom = true;
        }
    }

    ImGui::End();
}



void Interface::ShowDicomViewer() {
    // Only reload DICOM tag entries if the file changes
    static std::string lastDicomPath="";
    static std::vector<DicomEntry> entries;
    const auto& dicomFilePaths = this->dcmReader.getDicomFilePaths();

    if (dicomFilePaths !="") {
        
        if (dicomFilePaths != lastDicomPath) {
            entries = loadDicomTags(dicomFilePaths);
            lastDicomPath = dicomFilePaths;
        }
    }

    static char groupInput[5] = "";
    static char elementInput[5] = "";
    static std::string searchResult = "";
    static bool shouldScrollTagList = false;
    static bool shouldScrollSearchResult = false;

    ImGui::Begin("DICOM Viewer", nullptr, ImGuiWindowFlags_NoCollapse);

    // --- Search Tag Input Fields & Button ---
    ImGui::Text("Search Tag:");
    float inputFieldWidth = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2;
    ImGui::SetNextItemWidth(inputFieldWidth);
    ImGui::InputTextWithHint("##Group", "Group (hex)", groupInput, sizeof(groupInput), ImGuiInputTextFlags_CharsHexadecimal);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(inputFieldWidth);
    ImGui::InputTextWithHint("##Element", "Element (hex)", elementInput, sizeof(elementInput), ImGuiInputTextFlags_CharsHexadecimal);

    // Search Button - always visible below inputs, takes full width
    if (ImGui::Button("Search", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
        uint16_t group = 0, element = 0;
        std::stringstream ss1, ss2;
        ss1 << std::hex << groupInput;
        ss1 >> group;
        ss2 << std::hex << elementInput;
        ss2 >> element;

        bool found = false;
        for (const auto& entry : entries) {
            if (entry.tag.group() == group && entry.tag.element() == element) {
                std::stringstream tagStr;
                tagStr << std::hex << std::uppercase << std::setfill('0')
                       << "(" << std::setw(4) << group << "," << std::setw(4) << element << ")";
                searchResult = tagStr.str() + " | " + entry.tagName + ": " + entry.value;
                found = true;
                break;
            }
        }
        if (!found) {
            std::stringstream tagStr;
            tagStr << std::hex << std::uppercase << std::setfill('0')
                   << "(" << std::setw(4) << group << "," << std::setw(4) << element << ")";
            searchResult = tagStr.str() + ": Not Found";
        }
        shouldScrollSearchResult = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // --- Search Result Area (fixed height, always visible) ---
    ImGui::Text("Search Result:");
    float searchResultFixedH = ImGui::GetTextLineHeightWithSpacing() * 2 + ImGui::GetStyle().ItemSpacing.y * 2;
    ImGui::BeginChild("searchResultChild", ImVec2(0, searchResultFixedH), true, ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::TextWrapped("%s", searchResult.c_str());
    if (shouldScrollSearchResult) {
        ImGui::SetScrollHereY(1.0f);
        shouldScrollSearchResult = false;
    }
    ImGui::EndChild();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // --- Tag List Area (takes remaining available space) ---
    ImGui::Text("Tags:");
    // ImVec2(0,0) makes this child take up the rest of the available height in the parent window.
    ImGui::BeginChild("tagListChild", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::Spacing();
    for (size_t i = 0; i < std::min<size_t>(20, entries.size()); ++i) {
        const auto& entry = entries[i];
        ImGui::TextWrapped("(%04X,%04X) | %-30s: %s",
            entry.tag.group(), entry.tag.element(),
            entry.tagName.c_str(), entry.value.c_str());
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
    }
    if (shouldScrollTagList) {
        ImGui::SetScrollHereY(1.0f);
        shouldScrollTagList = false;
    }
    ImGui::EndChild();

    ImGui::End();
}


