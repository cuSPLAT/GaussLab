// model.hpp
#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <torch/torch.h>
#include "includes/dicom_loader.hpp"
#include "knn.hpp"

using namespace torch::indexing;
torch::Tensor rgb2sh(const torch::Tensor &rgb);

struct PlyData
{
  torch::Tensor means;
  torch::Tensor featuresDc;
  torch::Tensor opacities;
  torch::Tensor scales;
  torch::Tensor quats;
};

struct SceneBE
{
  std::unique_ptr<float[]> sceneDataBuffer;
  size_t verticesCount = 0;
  size_t bufferSize = 0;
  float centroid[3] = {0.0f, 0.0f, 0.0f};
};

struct Model
{
  Model(const InputData &inputData,
        const torch::Device &device):
    device(device)
    {
      long long numPoints = inputData.points.xyz.size(0);

      torch::manual_seed(42);

      means = inputData.points.xyz.to(device);

      torch::Tensor initial_scales = spatial_grid_knn_scales(means, 4); // 4 = self + 3 neighbors
      scales = initial_scales.repeat({1, 3}).log();
      
      torch::Tensor identity_quat = torch::tensor({0.0f, 0.0f, 0.0f, 1.0f}, 
                                    torch::dtype(torch::kFloat32).device(device));
      quats = identity_quat.repeat({numPoints, 1});

      // featuresDc = rgb2sh(inputData.points.rgb.toType(torch::kFloat64) / 255.0).toType(torch::kFloat32).to(device);
      featuresDc = rgb2sh(inputData.points.rgb).to(device);
      opacities = torch::logit(0.1f * torch::ones({numPoints, 1})).to(device);
      backgroundColor = torch::zeros({1}, device);
  }

  PlyData getPlyData();
  // void savePly(const std::string &filename);
  // void savePlyOptimized(const std::string &filename);
  
  torch::Tensor means;
  torch::Tensor scales;
  torch::Tensor quats;
  torch::Tensor featuresDc;
  torch::Tensor opacities;
  
  torch::Tensor radii; // set in forward()
  torch::Tensor xys;   // set in forward()
  
  torch::Tensor backgroundColor;
  torch::Device device;

  size_t bufferSize;
};

void savePly(const std::string &filename, const PlyData& data);
SceneBE createSceneFromPlyData(const PlyData& data);

#endif
