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

    glm::vec3 centroid {0}; // a temporary

    // The main initial viewport
    Viewport::newViewport(width, height);

    while (!glfwWindowShouldClose(window)) {
        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        //ImGui::ShowMetricsWindow();

        createMenuBar();
        createDockSpace();
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::SetNextWindowDockID(dockspace_id, ImGuiCond_FirstUseEver);

        // ----------------------------- Drawing ------------------------
        Viewport::drawViewports_ImGui(renderer);
        Viewport& selectedViewport = Viewport::viewports[::globalState.selectedViewport];

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
                    ImGui::Text("Camera Position:");
                    ImGui::InputFloat3("Position", glm::value_ptr(selectedViewport.view_camera->cameraPos));

                    // Point cloud or Gaussian splatting view mode selection
                    if (ImGui::RadioButton("Scene", selectedViewport.view_camera->scene))
                        selectedViewport.view_camera->scene = true;
                    ImGui::SameLine();
                    if (ImGui::RadioButton("Object", !selectedViewport.view_camera->scene))
                        selectedViewport.view_camera->scene = false;
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
                    ImGui::Dummy(ImVec2(0.f, 10.f));
                    ImGui::SeparatorText("Gaussian Splats");
                    ImGui::Checkbox("Splat Sorting", &globalState.sortingEnabled);

                    // -------------------- Toolbox----------------------
                    ImGui::Dummy(ImVec2(0.f, 10.f));
                    ImGui::SeparatorText("Tool Box");
                    Tools::drawToolBox_ImGui();
                    ImGui::EndTabItem();
                }
            }
            ImGui::EndTabBar();
            
        }
        ImGui::End();

        // --------------------- Marching Cubes -----------------
        static bool rendered = false;
        if (ImGui::Begin("Marching Cubes")) {
            if (ImGui::SliderInt("Threads", &selected_index, 0, log2_threads - 1, ""))
                n_threads = allowed_threads[selected_index];
            ImGui::SameLine();
            ImGui::Text("%d", allowed_threads[selected_index]);
            if(ImGui::Button("March")) {
                if (dcmReader.loadedData.readable.test()) {
                    rendered = false;
                    const DicomReader::DicomData& data = dcmReader.loadedData;

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
            if (MarchingCubes::marched.test() && !rendered) {
                rendered = true;
                // for debugging
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duration = now - MarchingCubes::last_iter_timer;
                mc_duration = duration.count();

                renderer->constructMeshScene(MarchingCubes::OutputVertices);

                for (int i = 0; i < Viewport::n_viewports; i++)
                    Viewport::viewports[i].view_camera->setCentroid(centroid);
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
