# CMakeLists.txt
cmake_minimum_required(VERSION 3.21)
project(cusplat)

set(GPU_RUNTIME "CUDA" CACHE STRING "CUDA or CPU")
set(MAX_CUDA_COMPATIBILITY OFF CACHE BOOL "Build for maximum CUDA device compatibility")
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
# Use time-of-extraction for FetchContent'ed files modification time
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()
# Suppress warning #20012-D (nvcc and glm)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -diag-suppress=20012)
set(GCC_OPENMP_INCLUDE_DIR "/usr/lib/gcc/x86_64-unknown-linux-gnu/14.2/include")

include(FetchContent)

FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.2.0.zip
)
FetchContent_MakeAvailable(cxxopts)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# -DNDEBUG:   Disable asserts and other debugging code
# -march=native: Generate code optimized for the specific CPU architecture of the build machine
# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native")
#     message(STATUS "Added GCC/Clang release optimizations: ${CMAKE_CXX_FLAGS_RELEASE}")
# endif()

if(GPU_RUNTIME STREQUAL "CUDA")
    find_package(CUDAToolkit)
    if (NOT CUDAToolkit_FOUND)
        message(WARNING "CUDA toolkit not found, building with CPU support only")
        set(GPU_RUNTIME "CPU")
    else()
        if (MAX_CUDA_COMPATIBILITY)
            execute_process(COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --list-gpu-arch 
                    OUTPUT_VARIABLE LIST_GPU_ARCH 
                    ERROR_QUIET)
        endif()

        if(NOT LIST_GPU_ARCH AND MAX_CUDA_COMPATIBILITY)
            message(WARNING "Cannot compile for max CUDA compatibility, nvcc does not support --list-gpu-arch")
            SET(MAX_CUDA_COMPATIBILITY OFF)
        endif()
        if(NOT MAX_CUDA_COMPATIBILITY)
            if(NOT CMAKE_CUDA_ARCHITECTURES)
                SET(CMAKE_CUDA_ARCHITECTURES 75)
                # SET(CMAKE_CUDA_ARCHITECTURES 70;75;80)
            endif()
        else()
            # Build for maximum compatibility
            # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
            set(CMAKE_CUDA_ARCHITECTURES "")

            # Extract list of arch and gencodes
            string(REPLACE "\r" "" LIST_GPU_ARCH ${LIST_GPU_ARCH})
            string(REPLACE "\n" ";" LIST_GPU_ARCH ${LIST_GPU_ARCH})

            execute_process(COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --list-gpu-code 
                OUTPUT_VARIABLE LIST_GPU_CODE 
                ERROR_QUIET)
            string(REPLACE "\r" "" LIST_GPU_CODE ${LIST_GPU_CODE})
            string(REPLACE "\n" ";" LIST_GPU_CODE ${LIST_GPU_CODE})

            list(GET LIST_GPU_CODE 0 TARGET_GPU_CODE)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${TARGET_GPU_CODE}")

            set(IDX 0)
            foreach(GPU_ARCH ${LIST_GPU_ARCH})
                string(REGEX MATCH "compute_([0-9]+)" GPU_ARCH_VERSION "${GPU_ARCH}")
                list(APPEND CMAKE_CUDA_ARCHITECTURES "${CMAKE_MATCH_1}")
                list(GET LIST_GPU_CODE ${IDX} GPU_CODE)
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=${GPU_ARCH},code=${GPU_CODE}")
                math(EXPR IDX "${IDX}+1")
            endforeach()
            message("Set CUDA flags: " ${CMAKE_CUDA_FLAGS})
        endif()
        # Set torch cuda architecture list
        set(TORCH_CUDA_ARCH_LIST ${CMAKE_CUDA_ARCHITECTURES})
        list(TRANSFORM TORCH_CUDA_ARCH_LIST REPLACE "([0-9])([0-9])" "\\1.\\2")
        string(REPLACE ";" " " TORCH_CUDA_ARCH_LIST "${TORCH_CUDA_ARCH_LIST}")
        message(STATUS "** Updated TORCH_CUDA_ARCH_LIST to ${TORCH_CUDA_ARCH_LIST} **")
    endif()
else()
    set(GPU_RUNTIME "CPU")
    message(STATUS "etet")
endif()

set(CMAKE_CXX_STANDARD 17)
if(GPU_RUNTIME STREQUAL "CUDA")
    enable_language(${GPU_RUNTIME})
    set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
    set(${GPU_RUNTIME}_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # For GCC (which emits this warning) and Clang (which might have similar ones or respect GCC flags)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow")
    message(STATUS "Suppressing -Wstringop-overflow for C++ files.")
endif()

if (NOT WIN32 AND NOT APPLE)
    set(STDPPFS_LIBRARY stdc++fs)
endif()

find_package(Torch REQUIRED)

if (NOT WIN32 AND NOT APPLE)
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
endif()

set(GSPLAT_LIBS gsplat)
if(GPU_RUNTIME STREQUAL "CUDA")
    add_library(gsplat gsplat/forward.cu gsplat/bindings.cu gsplat/helpers.cuh)
    # list(APPEND GSPLAT_LIBS gsplat)
    if(GPU_RUNTIME STREQUAL "CUDA")
        set(GPU_LIBRARIES "cuda")
        target_link_libraries(gsplat PUBLIC cuda)
    endif()
    target_include_directories(gsplat PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${TORCH_INCLUDE_DIRS}
    )
    #target_link_libraries(gsplat PUBLIC glm::glm-header-only)
    set_target_properties(gsplat PROPERTIES LINKER_LANGUAGE CXX)
endif()

set(DCMCORE_FILES
    dcmcore/binary_file.cpp
    dcmcore/data_dictionary.cpp
    dcmcore/data_element.cpp
    dcmcore/data_set.cpp
    dcmcore/defs.cpp
    dcmcore/dicom_reader.cpp
    dcmcore/read_handler.cpp
    dcmcore/tag.cpp
    dcmcore/tag_table.cpp
    dcmcore/util.cpp
    dcmcore/visitor.cpp
    dcmcore/vr.cpp
)
set(SRC_FILES 
    knn.cu
    dicom_loader.cpp
    engine.cpp
    model.cpp
    #main.cpp
)

add_library(${PROJECT_NAME} STATIC ${DCMCORE_FILES} ${SRC_FILES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${GPU_INCLUDE_DIRS}
    ${GCC_OPENMP_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${STDPPFS_LIBRARY}
    ${GPU_LIBRARIES}
    gsplat
    ${TORCH_LIBRARIES}
)
if (Pangolin_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${Pangolin_LIBRARIES})
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE
    cxxopts::cxxopts
)
if (NOT WIN32)
    target_link_libraries(${PROJECT_NAME} PUBLIC pthread)
endif()

if(GPU_RUNTIME STREQUAL "CUDA")
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA)
endif()

# Apply OpenMP settings to the library
message(STATUS "Manually configuring OpenMP with -fopenmp for ${PROJECT_NAME} library.")
target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.13")
    target_link_options(${PROJECT_NAME} PRIVATE -fopenmp)
else()
    set_property(TARGET ${PROJECT_NAME} APPEND_STRING PROPERTY LINK_FLAGS " -fopenmp")
endif()
