#ifndef KNN_H
#define KNN_H

#include <torch/torch.h>

// Computes the average distance to the k-1 nearest neighbors for each point.
// @param points A (N, 3) float tensor of 3D points, must be on the GPU.
// @param k The number of neighbors to find (e.g., 4 for the point itself + 3 neighbors).
// @return A (N, 1) float tensor containing the calculated scale for each point.
torch::Tensor spatial_grid_knn_scales(const torch::Tensor& points, int k);

#endif
