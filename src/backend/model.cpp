// model.cpp
#include <ATen/core/interned_strings.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include "includes/model.hpp"
#include "includes/dicom_loader.hpp"
#include "includes/engine.hpp"
#include <thread>
#include <omp.h>

namespace fs = std::filesystem;

torch::Tensor randomQuatTensor(long long n)
{
    torch::Tensor u = torch::rand(n);
    torch::Tensor v = torch::rand(n);
    torch::Tensor w = torch::rand(n);
    return torch::stack({
        torch::sqrt(1 - u) * torch::sin(2 * PI * v),
        torch::sqrt(1 - u) * torch::cos(2 * PI * v),
        torch::sqrt(u) * torch::sin(2 * PI * w),
        torch::sqrt(u) * torch::cos(2 * PI * w)
    }, -1);
}

SceneBE createSceneFromPlyData(const PlyData& data)
{
    SceneBE scene;

    scene.verticesCount = data.means.size(0);
    if (scene.verticesCount == 0) { return scene; }
    int32_t dcF = data.featuresDc.size(1);

    // --- 1) Calculate Centroid ---
    torch::Tensor centroidTensor = torch::mean(data.means, /*dim=*/0);
    memcpy(scene.centroid, centroidTensor.data_ptr<float>(), 3 * sizeof(float));

    // --- 2) Prepare for buffer creation ---
    size_t floats_per_pt = 3  /*xyz*/ + 3 /*colors*/
                          + 1 /*opacity*/ + 3 /*scale*/ + 4 /*quat*/;
    size_t record_bytes = floats_per_pt * sizeof(float);
    scene.bufferSize = record_bytes * scene.verticesCount;
    // Allocate the single, contiguous buffer
    scene.sceneDataBuffer = std::make_unique<float[]>(floats_per_pt * scene.verticesCount);
    // TODO: there is no need to re-init but for some reason it errors
    // most probably an include mismatch
    torch::Tensor scales = torch::exp(data.scales);
    torch::Tensor opacities = 1 / (1 + torch::exp(-data.opacities));

    // --- 3) Get raw data pointers from CPU tensors once ---
    const float* means_ptr     = data.means.data_ptr<float>();
    const float* features_ptr  = data.featuresDc.data_ptr<float>();
    const float* opacities_ptr = opacities.data_ptr<float>();
    const float* scales_ptr    = scales.data_ptr<float>();
    const float* quats_ptr     = data.quats.data_ptr<float>();

    size_t current_offset = 0;
    std::memcpy(
        scene.sceneDataBuffer.get() + current_offset, means_ptr, data.means.size(0) * 3 * sizeof(float)
    );
    current_offset += data.means.size(0) * 3;
    std::memcpy(
        scene.sceneDataBuffer.get() + current_offset, features_ptr, data.featuresDc.size(0) * 3 * sizeof(float)
    );
    current_offset += data.featuresDc.size(0) * 3;
    std::memcpy(
        scene.sceneDataBuffer.get() + current_offset, opacities_ptr, data.opacities.size(0) * sizeof(float)
    );
    current_offset += data.opacities.size(0);
    std::memcpy(
        scene.sceneDataBuffer.get() + current_offset, scales_ptr, data.scales.size(0) * 3 * sizeof(float)
    );
    current_offset += data.scales.size(0) * 3;
    std::memcpy(
        scene.sceneDataBuffer.get() + current_offset, quats_ptr, data.quats.size(0) * 4 * sizeof(float)
    );

    return scene;
}

torch::Tensor Model::forward(CameraYassa& cam)
{
    const float fx = cam.fx;
    const float fy = cam.fy;
    const float cx = cam.cx;
    const float cy = cam.cy;
    const int height = static_cast<int>(static_cast<float>(cam.height));
    const int width = static_cast<int>(static_cast<float>(cam.width));

    torch::Tensor R_mat = cam.camToWorld.index({Slice(None, 3), Slice(None, 3)});
    torch::Tensor T_vec = cam.camToWorld.index({Slice(None, 3), Slice(3,4)});

    // Flip the z and y axes to align with gsplat conventions
    R_mat = torch::matmul(R_mat, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f}, R_mat.device())));

    // worldToCam
    torch::Tensor Rinv = R_mat.transpose(0, 1);
    torch::Tensor Tinv = torch::matmul(-Rinv, T_vec);

    torch::Tensor viewMat = torch::eye(4, device);
    viewMat.index_put_({Slice(None, 3), Slice(None, 3)}, Rinv);
    viewMat.index_put_({Slice(None, 3), Slice(3, 4)}, Tinv);
        
    float fovX = 2.0f * std::atan(width / (2.0f * fx));
    float fovY = 2.0f * std::atan(height / (2.0f * fy));

    float tanHalfFovX = std::tan(0.5f * fovX);
    float tanHalfFovY = std::tan(0.5f * fovY);
    torch::Tensor projMat = torch::zeros({4, 4}, torch::dtype(torch::kFloat32).device(device));
    projMat.index_put_({0, 0}, 1.0f / tanHalfFovX);
    projMat.index_put_({1, 1}, 1.0f / tanHalfFovY);
    projMat.index_put_({2, 2}, 1.0f );
    projMat.index_put_({3, 2}, 1.0f);

    TileBounds tileBounds = std::make_tuple((width + BLOCK_X - 1) / BLOCK_X,
                    (height + BLOCK_Y - 1) / BLOCK_Y,
                    1);

    auto p_tuple = custom_ops::ProjectGaussians::forward(means,
        torch::exp(scales),
        1.0f, 
        quats / quats.norm(2, {-1}, true),
        viewMat,
        torch::matmul(projMat, viewMat),
        fx, fy, cx, cy,
        height, width,
        tileBounds,
        0.01f);

    xys = std::get<0>(p_tuple);
    torch::Tensor depths_from_proj = std::get<1>(p_tuple);
    radii = std::get<2>(p_tuple);
    torch::Tensor conics_from_proj = std::get<3>(p_tuple);
    torch::Tensor numTilesHit_from_proj = std::get<4>(p_tuple);

    if (radii.sum().item<float>() == 0.0f)
        return backgroundColor.repeat({height, width, 1});

    torch::Tensor gaussian_colors = sh2rgb(featuresDc);

    torch::Tensor rgb_output = custom_ops::RasterizeGaussians::forward(
            xys,
            depths_from_proj,
            radii,
            conics_from_proj,
            numTilesHit_from_proj,
            gaussian_colors,
            torch::sigmoid(opacities),
            height,
            width,
            backgroundColor);

    rgb_output = torch::clamp_max(rgb_output, 1.0f);

    return rgb_output;
}

PlyData Model::getPlyData()
{
    PlyData data;
    data.means      = means.cpu();
    data.featuresDc = featuresDc.cpu();
    data.opacities  = opacities.cpu();
    data.scales     = scales.cpu();
    data.quats      = quats.cpu();
    return data;
}

void savePly(const std::string &filename, const PlyData& data)
{
    SceneBE scene = createSceneFromPlyData(data);

    if (scene.verticesCount == 0)
    {
        std::cerr << "Warning: Attempting to save an empty scene." << std::endl;
        return;
    }
    int32_t dcF = data.featuresDc.size(1);

    // ASCII header
    std::ostringstream hdr;
    hdr << "ply\n"
        << "format binary_little_endian 1.0\n"
        << "comment Generated by cusplat medical pipeline\n"
        << "comment centroid_x " << scene.centroid[0] << "\n"
        << "comment centroid_y " << scene.centroid[1] << "\n"
        << "comment centroid_z " << scene.centroid[2] << "\n"
        << "element vertex " << scene.verticesCount << "\n" // Uses scene.verticesCount
        << "property float x\nproperty float y\nproperty float z\n"
        << "property float nx\nproperty float ny\nproperty float nz\n";
    for (int i = 0; i < dcF;   ++i) hdr << "property float f_dc_"   << i << "\n";
    hdr << "property float opacity\n"
        << "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        << "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        << "end_header\n";

    std::ofstream o(filename, std::ios::binary);
    auto const &hs = hdr.str();
    o.write(hs.data(), hs.size());
    o.write(reinterpret_cast<const char*>(scene.sceneDataBuffer.get()), scene.bufferSize);
    o.close();
    std::cout << "Wrote " << filename << std::endl;
}
