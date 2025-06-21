cmake_minimum_required(VERSION 3.20)

project(GausStudio VERSION 0.1 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures to build for" FORCE)
message(STATUS "[GausStudio] Setting global CMAKE_CUDA_ARCHITECTURES to: ${CMAKE_CUDA_ARCHITECTURES}")

# Propagate this setting to PyTorch for its JIT compiler (important!)
set(TORCH_CUDA_ARCH_LIST "7.5" CACHE STRING "PyTorch JIT CUDA arch list" FORCE)
message(STATUS "[GausStudio] Setting TORCH_CUDA_ARCH_LIST to: ${TORCH_CUDA_ARCH_LIST}")

# set(CMAKE_PREFIX_PATH external/libtorch)

add_subdirectory(external/nativefiledialog)
add_subdirectory(external/vega)
add_subdirectory(src/backend)

find_package(OpenGL REQUIRED)
find_package(Torch REQUIRED)
find_package(glfw3 3.4 REQUIRED)
find_package(glm REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
SET(CMAKE_CUDA_ARCHITECTURES 86)

set(IMGUI_SOURCES
    "external/imgui/imgui.cpp"
    "external/imgui/imgui_draw.cpp"
    "external/imgui/imgui_demo.cpp"
    "external/imgui/imgui_tables.cpp"
    "external/imgui/imgui_widgets.cpp"
    "external/imgui/backends/imgui_impl_glfw.cpp"
    "external/imgui/backends/imgui_impl_opengl3.cpp"
)

set(INTERFACE_SRC
    "src/interface/callbacks.cpp"
    "src/interface/main_interface.cpp"
    "src/interface/viewport.cpp"
)

set(CORE
    "src/core/renderer.cpp"
    "src/core/shaders.cpp"
    "src/core/camera.cpp"
    "src/core/scene_loader.cpp"
)

set(TOOLS
    "src/tools/tools.cpp"
    "src/tools/meshslicing.cpp"
)

set(CUDA_UTILS
    "src/cuda/render_utils.cu"
)

set(ALGORITHMS
    "src/algorithms/marchingcubes.cpp"
)

set(DATA_READER
    "src/data_reader/dicom_reader.cpp"
)

include_directories(SYSTEM "external/glad/include")
include_directories(SYSTEM "external/imgui")
include_directories(SYSTEM "external/happly")
include_directories(SYSTEM "external/nativefiledialog/src/include")
include_directories(SYSTEM "external/imgui/backends")
include_directories(SYSTEM "external/vega/include")
include_directories(SYSTEM "external/RadixSort")

add_executable(
    GausStudio
    "src/main.cpp"
    "src/debug_utils.cpp"
    ${PROJECT_SOURCES}
    ${IMGUI_SOURCES}
    ${INTERFACE_SRC}
    ${CORE}
    ${TOOLS}
    ${ALGORITHMS}
    ${DATA_READER}
    ${CUDA_UTILS}
    "external/glad/src/glad.c"
)
target_include_directories(GausStudio
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(GausStudio PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
message(STATUS "[GausStudio] Forcing GausStudio CUDA arch to: ${CMAKE_CUDA_ARCHITECTURES}")

target_link_libraries(GausStudio "${TORCH_LIBRARIES}")
target_link_libraries(GausStudio Threads::Threads)

# 7.5 is for turing architecture (GTX16)
# 8.6 is for ampere (RTX)
#set_target_properties(GausStudio PROPERTIES CUDA_ARCHITECTURES "70;75;86")
set_target_properties(GausStudio PROPERTIES CUDA_ARCHITECTURES 86)

target_link_libraries(
    GausStudio
    OpenGL::GL
    glfw
    glm::glm

    nfd
    vega
    cusplat
    -fopenmp
)
