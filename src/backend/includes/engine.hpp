// engine.hpp
#ifndef ENGINE_H
#define ENGINE_H

#include <gsplat/config.h>
#include <gsplat/bindings.h>

#include <torch/torch.h>
#include <tuple>
#include <string>
 
#include <torch/torch.h>
typedef std::tuple<int, int, int> TileBounds;

// ###############################################################################
// project_gaussians 
// ###############################################################################
namespace custom_ops
{
    struct ProjectGaussians
    {
        static std::tuple<torch::Tensor, // xys
                          torch::Tensor, // depths
                          torch::Tensor, // radii
                          torch::Tensor, // conics
                          torch::Tensor, // numTilesHit
                          torch::Tensor> // cov3d
        forward(
            torch::Tensor means,
            torch::Tensor scales,
            float globScale,
            torch::Tensor quats,
            torch::Tensor viewMat,
            torch::Tensor projMat,
            float fx, float fy, float cx, float cy,
            int imgHeight, int imgWidth,
            TileBounds tileBounds,
            float clipThresh = 0.01f); // Added 'f'
    };
    
    struct RasterizeGaussians
    {
        static torch::Tensor forward(
            torch::Tensor xys,
            torch::Tensor depths,
            torch::Tensor radii,
            torch::Tensor conics,
            torch::Tensor numTilesHit,
            torch::Tensor colors,
            torch::Tensor opacity,
            int imgHeight, int imgWidth,
            torch::Tensor background);
    };
    
} // namespace custom_ops

// ###############################################################################
// sperical_harmonics
// ###############################################################################
torch::Tensor rgb2sh(const torch::Tensor &rgb);
torch::Tensor sh2rgb(const torch::Tensor &sh);

// ###############################################################################
// rasterize_gaussians
// ###############################################################################
std::tuple<torch::Tensor,
           torch::Tensor> binAndSortGaussians(int numPoints, 
                                              int numIntersects,
                                              torch::Tensor xys,
                                              torch::Tensor depths,
                                              torch::Tensor radii,
                                              torch::Tensor cumTilesHit,
                                              TileBounds tileBounds);

#endif 