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
find_package(glfw3 3.3.10 REQUIRED)
find_package(glm REQUIRED)
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(DCMTK REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(STB_IMAGE_IMPLEMENTATION ON)
add_definitions(-DSTB_IMAGE_IMPLEMENTATION)

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
    "src/interface/dicom_viewer.cpp"
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

set(ALGORITHMS
    "src/algorithms/marchingcubes.cpp"
)

set(DATA_READER
    "src/data_reader/dicom_reader.cpp"
)

set(STB_IMPL
    "src/stb_image_write_impl.cpp"
)

include_directories(SYSTEM "external/glad/include")
include_directories(SYSTEM "external/imgui")
include_directories(SYSTEM "external/happly")
include_directories(SYSTEM "external/nativefiledialog/src/include")
include_directories(SYSTEM "external/imgui/backends")
include_directories(SYSTEM "external/vega/include")
include_directories(SYSTEM "external/RadixSort")
include_directories(/usr/local/cuda/include)
include_directories(SYSTEM "external/ImGuiFileDialog")
include_directories(${DCMTK_INCLUDE_DIRS})

set(SOURCES
    src/main.cpp
    src/interface/main_interface.cpp
    src/interface/dicom_viewer.cpp
    src/core/renderer.cpp
    src/core/scene_loader.cpp
    src/algorithms/marchingcubes.cpp
    src/data_reader/dicom_reader.cpp
    src/tools/tools.cpp
    src/stb_image_write_impl.cpp
)

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
    ${STB_IMPL}
    "external/glad/src/glad.c"
    "external/ImGuiFileDialog/ImGuiFileDialog.cpp"
)
target_include_directories(GausStudio
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(GausStudio "${TORCH_LIBRARIES}")
target_link_libraries(GausStudio Threads::Threads)
target_link_libraries(GausStudio ${DCMTK_LIBRARIES})

# 7.5 is for turing architecture (GTX16)
# 8.6 is for ampere (RTX)
set_target_properties(GausStudio PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_link_libraries(
    GausStudio
    OpenGL::GL
    glfw
    glm::glm
    CURL::libcurl
    nfd
    vega
)
