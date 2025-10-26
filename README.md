<a id="readme-top"></a>
<div align="center">
  <h1 align="center" style="font-size: 60px;">GaussLab</h1>
  <p align="center">
        A fully featured studio for reconstructing CT-scans using Marching Cubes and Gaussian Splatting
  </p>
  <img src="assets/demo_gif.gif" alt="animated" />
</div>

# About
GaussLab is a fully-featured studio for reconstructing CT-scans using classical approaches like Marching Cubes and SOTA approaches, Gaussian Splatting. GaussLab is fully GPU-accelerated

# Key-features
1. 3D reconstruction using Marching Cubes and Gaussian Splatting
2. Multi-viewport viewing
3. AI Assistant for Diagnosis
4. Object slicing

# Building
GaussLab uses the CMake build system, install the following prerequisites first:
- glm
- LibTorch, PyTorch's C++ frontend

Then create a build directory and run
```shell
cmake ..
make
```

## Planned features
- [ ] CUDA backend for rasterization
- [ ] Segmentation models
