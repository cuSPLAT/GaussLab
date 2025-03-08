cmake_minimum_required(VERSION 3.20)
project(Tigre LANGUAGES CXX CUDA)

add_library(
    Tigre 
    "GpuIds.cpp"
    "TIGRE_common.cpp"
    "Siddon_projection.cu"
    "ray_interpolated_projection.cu"
)
