cmake_minimum_required(VERSION 3.20)

project(GausStudio VERSION 0.1 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(CMAKE_PREFIX_PATH external/libtorch)

add_subdirectory(external/nativefiledialog)
add_subdirectory(external/vega)

find_package(OpenGL REQUIRED)
find_package(Torch REQUIRED)
find_package(glfw3 3.4 REQUIRED)
find_package(glm REQUIRED)
find_package(Threads REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(IMGUI_SOURCES
    "external/imgui/imgui.cpp"
    "external/imgui/imgui_draw.cpp"
    "external/imgui/imgui_demo.cpp"
    "external/imgui/imgui_tables.cpp"
    "external/imgui/imgui_widgets.cpp"
    "external/imgui/backends/imgui_impl_glfw.cpp"
    "external/imgui/backends/imgui_impl_opengl3.cpp"

)

set(PROJECT_SOURCES 
    "src/main.cpp"
    "src/renderer.cpp"
    "src/shaders.cpp"
    "src/camera.cpp"
    "src/debug_utils.cpp"
    "src/main_interface.cpp"
    "src/scene_loader.cpp"
    "src/callbacks.cpp"
    "src/tools/tools.cpp"
    "src/tools/meshslicing.cpp"
    "src/data_reader/dicom_reader.cpp"
    "src/cuda/render_utils.cu"
    "src/algorithms/marchingcubes.cpp"
)

include_directories(SYSTEM "external/glad/include")
include_directories(SYSTEM "external/imgui")
include_directories(SYSTEM "external/happly")
include_directories(SYSTEM "external/nativefiledialog/src/include")
include_directories(SYSTEM "external/imgui/backends")
include_directories(SYSTEM "external/Tigre")
include_directories(SYSTEM "external/vega/include")
include_directories(SYSTEM "external/RadixSort")

add_executable(
    GausStudio
    ${PROJECT_SOURCES}
    ${IMGUI_SOURCES}
    "external/glad/src/glad.c"
)
target_include_directories(GausStudio PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(GausStudio "${TORCH_LIBRARIES}")
target_link_libraries(GausStudio Threads::Threads)

# 7.5 is for turing architecture (GTX16)
# 8.6 is for ampere (RTX)
set_target_properties(GausStudio PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_link_libraries(
    GausStudio
    OpenGL::GL
    glfw
    glm::glm

    nfd
    vega
)
