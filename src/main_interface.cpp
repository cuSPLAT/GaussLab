#include "main_interface.h"
#include "algorithms/marchingcubes.h"
#include "cuda/debug_utils.h"
#include "nfd.h"
#include "renderer.h"
#include "scene_loader.h"
#include "callbacks.h"

#include <chrono>
#include <cstddef>
#include <fstream>
#include <glm/fwd.hpp>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <nfd.hpp>
#include <nfd_glfw3.h>

#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui.h>

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
    glfwSetCursorPosCallback(window, Callbacks::mouse_callback);
    glfwSetScrollCallback(window, Callbacks::scroll_callback);

    return true;
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

    //Scene* pcd = PLYLoader::loadPLy("/home/Abdelrahman/Downloads/christmas_tree.ply");
    //pcd->centroid = centroid;
    //renderer->constructScene(pcd, vertices);
    
    //delete pcd;
}

void Interface::createViewWindow() {
    if (ImGui::Begin("View")) {
        ImVec2 viewSize = ImGui::GetWindowSize();
        ImGui::Image((ImTextureID)renderer->getRenderBuffer(), viewSize, ImVec2(0,1), ImVec2(1, 0));
    }
    ImGui::End();
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
    static glm::vec3 centroid; // a temporary

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
                    // --------------------- Marching Cubes -----------------
                    if (ImGui::SliderInt("Threads", &selected_index, 0, 2, ""))
                        n_threads = allowed_threads[selected_index];
                    ImGui::SameLine();
                    ImGui::Text("%d", allowed_threads[selected_index]);
                    if(ImGui::Button("March")) {
                        if (dcmReader.loadedData.readable.test()) {
                            DicomReader::DicomData& data = dcmReader.loadedData;
                            MarchingCubes::launchThreaded(
                                data.buffer.get(),
                                data.width, data.length, data.height,
                                660, centroid,
                                1, n_threads
                            );
                        }
                    }

                    if (MarchingCubes::marched.test()) {
                        MarchingCubes::marched.clear();
                        Scene scene;
                        renderer->constructScene(&scene, MarchingCubes::OutputVertices);
                    }
                    // ------------------------------------------------------
                    ImGui::EndTabItem();
                }
            }
            ImGui::EndTabBar();
        }
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

    if (ImGui::Begin("DockSpace Window", nullptr, window_flags)) {
        ImGui::PopStyleVar(2);

        // Define the dock space
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
    }
    ImGui::End();
}
