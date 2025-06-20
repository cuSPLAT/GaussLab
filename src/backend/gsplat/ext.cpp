#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // auto diff functions
    m.def("nd_rasterize_forward", &nd_rasterize_forward_tensor);
    m.def("rasterize_forward", &rasterize_forward_tensor);
    m.def("project_gaussians_forward", &project_gaussians_forward_tensor);
    // utils
    m.def("map_gaussian_to_intersects", &map_gaussian_to_intersects_tensor);
    m.def("get_tile_bin_edges", &get_tile_bin_edges_tensor);
}
