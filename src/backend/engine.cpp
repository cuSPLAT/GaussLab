// engine.cpp
#include "includes/engine.hpp"

namespace fs = std::filesystem;
using namespace torch::indexing;

// ###############################################################################
// project_gaussians
// ###############################################################################
namespace custom_ops
{
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ProjectGaussians::forward(
        torch::Tensor means,
        torch::Tensor scales,
        float globScale,
        torch::Tensor quats,
        torch::Tensor viewMat,
        torch::Tensor projMat,
        float fx, float fy, float cx, float cy,
        int imgHeight, int imgWidth,
        TileBounds tileBounds,
        float clipThresh)
    {
        int numPoints = means.size(0);
        auto t_kernel_output = project_gaussians_forward_tensor(numPoints, means, scales, globScale,
            quats, viewMat, projMat, fx, fy,
            cx, cy, imgHeight, imgWidth, tileBounds, clipThresh);

        torch::Tensor cov3d_out = std::get<0>(t_kernel_output);
        torch::Tensor xys_out = std::get<1>(t_kernel_output);
        torch::Tensor depths_out = std::get<2>(t_kernel_output);
        torch::Tensor radii_out = std::get<3>(t_kernel_output);
        torch::Tensor conics_out = std::get<4>(t_kernel_output);
        torch::Tensor numTilesHit_out = std::get<5>(t_kernel_output);

        return std::make_tuple(xys_out, depths_out, radii_out, conics_out, numTilesHit_out, cov3d_out);
    }
    
    torch::Tensor RasterizeGaussians::forward(
        torch::Tensor xys,
        torch::Tensor depths,
        torch::Tensor radii,
        torch::Tensor conics,
        torch::Tensor numTilesHit,
        torch::Tensor colors,
        torch::Tensor opacity,
        int imgHeight, int imgWidth,
        torch::Tensor background)
    {
        int numPoints = xys.size(0);

        TileBounds tileBounds = std::make_tuple(
            (imgWidth + BLOCK_X - 1) / BLOCK_X,
            (imgHeight + BLOCK_Y - 1) / BLOCK_Y,
            1
        );
        std::tuple<int, int, int> block = std::make_tuple(BLOCK_X, BLOCK_Y, 1);
        std::tuple<int, int, int> imgSize = std::make_tuple(imgWidth, imgHeight, 1);
        
        torch::Tensor cumTilesHit = torch::cumsum(numTilesHit, 0, torch::kInt32);
        int numIntersects = cumTilesHit[cumTilesHit.size(0) - 1].item<int>();
    
        auto b = binAndSortGaussians(numPoints, numIntersects, xys, depths, radii, cumTilesHit, tileBounds);
        torch::Tensor gaussianIdsSorted = std::get<0>(b);
        torch::Tensor tileBins = std::get<1>(b);
    
        auto t = rasterize_forward_tensor(tileBounds, block, imgSize, 
                                gaussianIdsSorted,
                                tileBins,
                                xys,
                                conics,
                                colors,
                                opacity,
                                background);
        torch::Tensor outImg = std::get<0>(t);    
        return outImg;
    }
    
} // namespace custom_ops

// ###############################################################################
// sperical_harmonics
// ###############################################################################
const double C0 = 0.28209479177387814;

torch::Tensor rgb2sh(const torch::Tensor &rgb)
{
    // Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    return (rgb - 0.5) / C0;
}

torch::Tensor sh2rgb(const torch::Tensor &sh)
{
    // Converts from 0th spherical harmonic coefficients to RGB values [0,1]
    return torch::clamp((sh * C0) + 0.5, 0.0f, 1.0f);
}

// ###############################################################################
// rasterize_gaussians
// ###############################################################################
std::tuple<torch::Tensor,
           torch::Tensor> binAndSortGaussians(int numPoints, int numIntersects,
                                            torch::Tensor xys,
                                            torch::Tensor depths,
                                            torch::Tensor radii,
                                            torch::Tensor cumTilesHit,
                                            TileBounds tileBounds)
{
    auto t = map_gaussian_to_intersects_tensor(numPoints, numIntersects, 
                                        xys, depths, radii, cumTilesHit, tileBounds);
    
    // unique IDs for each gaussian in the form (tile | depth id)
    torch::Tensor isectIds = std::get<0>(t);

    // Tensor that maps isect_ids back to cumHitTiles
    torch::Tensor gaussianIds = std::get<1>(t);
    
    auto sorted = torch::sort(isectIds);

    // sorted unique IDs for each gaussian in the form (tile | depth id)
    torch::Tensor isectIdsSorted = std::get<0>(sorted);
    torch::Tensor sortedIndices = std::get<1>(sorted);

    // sorted Tensor that maps isect_ids back to cumHitTiles
    torch::Tensor gaussianIdsSorted = torch::gather(gaussianIds, 0, sortedIndices);

    // range of gaussians hit per tile
    torch::Tensor tileBins = get_tile_bin_edges_tensor(numIntersects, isectIdsSorted);
    return std::make_tuple(gaussianIdsSorted, tileBins);
}