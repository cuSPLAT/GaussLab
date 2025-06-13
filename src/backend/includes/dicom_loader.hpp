#ifndef DICOM_LOADER_H
#define DICOM_LOADER_H

#include <string>
#include <torch/torch.h>

using namespace torch::indexing;

// ###############################################################################
// tensor_math
// ###############################################################################
#define PI 3.14159265358979323846
std::tuple<torch::Tensor, torch::Tensor, float> autoScaleAndCenterPoses(const torch::Tensor &poses);

// ###############################################################################
// input_data
// ###############################################################################
struct Camera
{
    int id = -1;
    int width = 0;
    int height = 0;
    float fx = 0;
    float fy = 0;
    float cx = 0;
    float cy = 0;

    torch::Tensor camToWorld;
    torch::Tensor image_tensor; // [H, W, 3] float tensor image data
    torch::Tensor K; // 3x3 intrinsics matrix
    std::string filePath = "";

    Camera(){};
    // This function creates the intrinsics matrix from fx, fy, cx, cy
    torch::Tensor getIntrinsicsMatrix()
    {
        return torch::tensor({{fx,   0.0f, cx},
                              {0.0f, fy,   cy},
                              {0.0f, 0.0f, 1.0f}}, torch::kFloat32);
    }
};

struct Points
{
    torch::Tensor xyz;
    torch::Tensor rgb;
};

struct InputData
{
    std::vector<Camera> cameras;
    float scale;
    torch::Tensor translation;
    Points points;
};

InputData inputDataFromDicom(const std::string& dicom_folder_path, 
                             double ww, 
                             double wc, 
                             int hu_threshold, 
                             int downsample,
                             int up_direction_choice);

#endif