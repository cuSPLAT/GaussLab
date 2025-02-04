cmake_minimum_required(VERSION 3.20)

project(GausStudio VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

add_subdirectory(external/nativefiledialog)

# for assimp
set(BUILD_SHARED_LIBS OFF)

find_package(OpenGL REQUIRED)
find_package(glfw3 3.4 REQUIRED)
find_package(glm REQUIRED)

include_directories(SYSTEM "external/glad/include")
include_directories(SYSTEM "external/imgui")
include_directories(SYSTEM "external/happly")
include_directories(SYSTEM "external/nativefiledialog/src/include")
include_directories(SYSTEM "external/imgui/backends")

add_executable(
   GausStudio
   "src/main.cpp"
   "src/renderer.cpp"
   "src/shaders.cpp"
   "src/camera.cpp"
   "src/main_interface.cpp"
   "src/scene_loader.cpp"
   "src/callbacks.cpp"
   "external/glad/src/glad.c"
   "external/imgui/imgui.cpp"
   "external/imgui/imgui_draw.cpp"
   "external/imgui/imgui_demo.cpp"
   "external/imgui/imgui_tables.cpp"
   "external/imgui/imgui_widgets.cpp"
   "external/imgui/backends/imgui_impl_glfw.cpp"
   "external/imgui/backends/imgui_impl_opengl3.cpp"
)

target_link_libraries(
    GausStudio
    OpenGL::GL
    glfw
    glm::glm

    nfd
)
