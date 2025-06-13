// main.cpp
#include <filesystem>
#include <cxxopts.hpp>
#include <chrono>

#include "includes/model.hpp"
#include "includes/engine.hpp"
#include "includes/dicom_loader.hpp"

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

namespace fs = std::filesystem;
using namespace torch::indexing;

cxxopts::ParseResult cmdLineArgs(int argc, char* argv[]);
void coutVerification(InputData inputData);

int main(int argc, char *argv[])
{
    torch::Device device = torch::kCUDA;
    cxxopts::ParseResult result = cmdLineArgs(argc, argv);

    const std::string dicomFolderPath = result["input"].as<std::string>();
    const std::string outputScene = result["output"].as<std::string>();
    const double WW = result["window_width"].as<double>();
    const double WC = result["window_center"].as<double>();
    const int POINT_CLOUD_DOWNSAMPLE = result["downsample"].as<int>();
    const double HU_THRESHOLD = result["threshold"].as<int>();
    const int up_choice = result["up"].as<int>();

    try
    {
        InputData inputData = inputDataFromDicom(dicomFolderPath, WW, WC, HU_THRESHOLD, POINT_CLOUD_DOWNSAMPLE, up_choice);

        coutVerification(inputData);
        
        Model model(inputData, device);
        Camera& cam = inputData.cameras[0];
        model.forward(cam);

        PlyData plyData = model.getPlyData();

        std::cout << "scene data buffer for viewer..." << std::endl;
        Scene scene = createSceneFromPlyData(plyData);
        std::cout << "  - Vertices: " << scene.verticesCount << std::endl;
        std::cout << "  - Buffer Size: " << scene.bufferSize / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  - Centroid: (" << scene.centroid[0] << ", " << scene.centroid[1] << ", " << scene.centroid[2] << ")" << std::endl;

        std::cout << "Saving the PLY ..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        savePly(outputScene, plyData);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "savePly took " << duration.count() << " seconds.\n";

    }
    catch(const std::exception &e)
    {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}

cxxopts::ParseResult cmdLineArgs(int argc, char* argv[])
{
    cxxopts::Options options("cusplat_medical", "DICOM-based 3D Gaussian Splats generator for medical imaging");
    options.add_options()
        ("i,input", "Path to the DICOM folder", cxxopts::value<std::string>())
        ("o,output", "Path where to save output scene", cxxopts::value<std::string>()->default_value("splat.ply"))
    
        ("WW,window_width", "Window Width (WW) for intensity windowing", cxxopts::value<double>()->default_value("2000.0"))
        ("WC,window_center", "Window Center (WC) for intensity windowing", cxxopts::value<double>()->default_value("500.0"))
        ("d,downsample", "Factor to downsample the point cloud (1 = no downsampling)", cxxopts::value<int>()->default_value("1"))
        ("HU,threshold", "Minimum Hounsfield Unit (HU) value", cxxopts::value<int>()->default_value("300"))

        ("up", "Set final orientation. Model's nose will point along the chosen axis.\n"
            "  1: face => +X \n"
            "  2: face => +Y \n"
            "  3: face => +Z \n",
            cxxopts::value<int>()->default_value("2"))
     
        ("h,help", "Print usage");
    
    options.parse_positional({ "input" });
    options.positional_help("[dicom project path]");
    cxxopts::ParseResult result;
    try{ result = options.parse(argc, argv); }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        exit(EXIT_FAILURE);
    }
    
    if (result.count("help") || !result.count("input"))
    {
        std::cout << options.help() << std::endl;
        exit(EXIT_SUCCESS);
    }
    return result;
}

void coutVerification(InputData inputData)
{
    std::cout << "\n--- verification ---" << std::endl;
    if (!inputData.cameras.empty())
    {
        const auto& first_cam = inputData.cameras[0];
        std::cout << "no of cameras: " << inputData.cameras.size() << std::endl;
        std::cout << "first camera dimensions: " << first_cam.width << "x" << first_cam.height << std::endl;
        std::cout << "first camera image tensor shape: " << first_cam.image_tensor.sizes() << std::endl;
        std::cout << "first camera image tensor dtype: " << first_cam.image_tensor.dtype() << std::endl;
        std::cout << "first camera pose (camToWorld):\n" << first_cam.camToWorld << std::endl;
    }
    if (inputData.points.xyz.defined())
    {
        std::cout << "point cloud XYZ tensor shape: " << inputData.points.xyz.sizes() << std::endl;
        std::cout << "point cloud RGB tensor shape: " << inputData.points.rgb.sizes() << std::endl;
    }
    std::cout << "inputData scale: " << inputData.scale << std::endl;
    std::cout << "inputData translation:\n" << inputData.translation << std::endl;
}