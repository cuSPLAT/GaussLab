// dicom_loader.cpp

#include "includes/dicom_loader.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <optional>
#include <array>
#include <cmath> // for tan()
#include <unordered_map>
#include <utility> 
#include <omp.h>

#include "dcmcore/dicom_reader.h"
#include "dcmcore/read_handler.h"
#include "dcmcore/data_set.h"
#include "dcmcore/buffer.h"
#include "includes/dicom_utils.h"

#include <torch/torch.h>

namespace fs = std::filesystem;
using namespace torch::indexing;

struct DicomSliceInfo
{
    fs::path file_path;
    double slice_location_z;
};

bool compareSlices(const DicomSliceInfo& a, const DicomSliceInfo& b)
{
    return a.slice_location_z < b.slice_location_z;
}

bool loadDicomFile(const fs::path& filePath, dcmcore::DataSet& dataSet)
{
    dataSet.Clear();
    dcmcore::FullReadHandler readHandler(&dataSet);
    dcmcore::DicomReader dicomReader(&readHandler);
    if (!dicomReader.ReadFile(filePath.string()))
    {
        std::cerr << "Warning: Failed to read: " << filePath.filename() << std::endl;
        return false;
    }
    if (dataSet.GetSize() == 0)
    {
        std::cerr << "Warning: Empty dataset after loading: " << filePath.filename() << std::endl;
        return false;
    }
    return true;
}

std::vector<DicomSliceInfo> scanSortDicomFiles(const fs::path& dicom_folder_path)
{
    std::vector<DicomSliceInfo> slice_infos;
    dcmcore::DataSet tempDataSet;
    for (const auto& entry : fs::directory_iterator(dicom_folder_path))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".dcm")
        {
            if (loadDicomFile(entry.path(), tempDataSet))
            {
                if (dcm::hasAttr(tempDataSet, DicomTags_NG::PixelData) && dcm::getImagePositionPatient(tempDataSet))
                {
                    slice_infos.push_back({entry.path(), (*dcm::getImagePositionPatient(tempDataSet))[2]});
                } 
                else 
                {
                    std::cerr << "Warning: Skipping " << entry.path().filename() << " - missing required tags." << std::endl;
                }
            }
        }
    }
    if (!slice_infos.empty())
    {
        std::sort(slice_infos.begin(), slice_infos.end(), compareSlices);
        std::cout << "Loaded and sorted " << slice_infos.size() << " DICOM slices." << std::endl;
    }
    return slice_infos;
}

torch::Tensor get_alignment_matrix(int up_direction_choice)
{
    switch (up_direction_choice)
    {
        case 1:
            return torch::tensor({{1, 0, 0}, {0, 0, -1}, {0, 1, 0}}, torch::kFloat32);
        
        case 2:
            return torch::tensor({{0, 0, -1}, {1, 0, 0}, {0, 1, 0}}, torch::kFloat32);
        
        case 3:
            return torch::tensor({{0, 1, 0}, {0, 0, -1}, {-1, 0, 0}}, torch::kFloat32);

        default:
            throw std::invalid_argument("Invalid up_direction_choice. Must be 1, 2, or 3.");
    }
}

std::tuple<torch::Tensor, torch::Tensor, float> autoScaleAndCenterPoses(const torch::Tensor &poses)
{
    // Center at mean
    torch::Tensor origins = poses.index({"...", Slice(None, 3), 3});
    torch::Tensor center = torch::mean(origins, 0);
    origins -= center;

    // Scale
    float f = 1.0f / torch::max(torch::abs(origins)).item<float>();
    origins *= f;
    
    torch::Tensor transformedPoses = poses.clone();
    transformedPoses.index_put_({"...", Slice(None, 3), 3}, origins);

    return std::make_tuple(transformedPoses, center, f);
}


InputData inputDataFromDicom(const std::string& dicom_folder_path_str,
                             double WW, 
                             double WC, 
                             int HU_THRESHOLD, 
                             int POINT_CLOUD_DOWNSAMPLE,
                             int up_direction_choice)
{
    fs::path dicom_folder_path(dicom_folder_path_str);
    if (!fs::is_directory(dicom_folder_path))
    {
        throw std::runtime_error("Error: DICOM folder does not exist: " + dicom_folder_path_str);
    }

    std::vector<DicomSliceInfo> slice_infos = scanSortDicomFiles(dicom_folder_path);
    if (slice_infos.empty())
    {
        throw std::runtime_error("No valid DICOM files found in " + dicom_folder_path_str);
    }

    InputData ret;
    std::vector<std::array<double, 3>> all_points_3d;
    std::vector<std::array<unsigned char, 3>> all_colors_8bit;

    // --- we need these for thread-safe accumulation --- ? you don't even need the cameras
    std::vector<CameraYassa> cameras_accumulator;
    std::vector<std::pair<int, torch::Tensor>> poses_accumulator;

    #pragma omp parallel
    {
        dcmcore::DataSet thread_local_dataSet;
        // each thread will have its own temporary vectors to avoid locking on every pixel
        std::vector<CameraYassa> local_cameras;
        std::vector<std::pair<int, torch::Tensor>> local_poses;


        std::vector<std::array<double, 3>> local_points;
        std::vector<std::array<unsigned char, 3>> local_colors;

        #pragma omp for schedule(dynamic) nowait
        for (size_t i = 0; i < slice_infos.size(); ++i)
        {
            const auto& slice_info = slice_infos[i];
            if (!loadDicomFile(slice_info.file_path, thread_local_dataSet)) continue;

            // --- 1. extract Metadata ---
            unsigned short W = dcm::getColumns(thread_local_dataSet);
            unsigned short H = dcm::getRows(thread_local_dataSet);
            auto ipp_opt = dcm::getImagePositionPatient(thread_local_dataSet);
            auto iop_opt = dcm::getImageOrientationPatient(thread_local_dataSet);
            auto ps_opt = dcm::getPixelSpacing(thread_local_dataSet);

            if (!W || !H || !ipp_opt || !iop_opt || !ps_opt)
            {
                #pragma omp critical
                std::cerr << "Skipping slice " << slice_info.file_path.filename() << " due to missing geometry tags." << std::endl;
                continue;
            }

            // --- 2. create Camera object ---
            CameraYassa cam;
            int current_id = static_cast<int>(i);
            cam.id = current_id;
            cam.width = W;
            cam.height = H;

            // --- 3. Calculate Fake Perspective Intrinsics ---
            // We must invent a field of view. 30 degrees is a reasonable guess.
            const float fov_y_degrees = 30.0f;
            const float fov_y_radians = fov_y_degrees * (M_PI / 180.0f);
            cam.fy = static_cast<float>(H) / (2.0f * tan(fov_y_radians / 2.0f));
            cam.fx = cam.fy; // assume square pixels for simplicity
            cam.cx = static_cast<float>(W) / 2.0f;
            cam.cy = static_cast<float>(H) / 2.0f;
            cam.K = cam.getIntrinsicsMatrix();

            // --- 4. Calculate Pose (camToWorld) ---
            const auto& iop = *iop_opt; // [Xx,Xy,Xz, Yx,Yy,Yz]
            const auto& ipp = *ipp_opt; // [Tx,Ty,Tz]
            const auto& ps = *ps_opt;   // {row_spacing, col_spacing}
            
            torch::Tensor R = torch::zeros({3, 3}, torch::kFloat32);
            auto R_acc = R.accessor<float, 2>();
            
            // X vector
            R_acc[0][0] = iop[0]; R_acc[1][0] = iop[1]; R_acc[2][0] = iop[2];
            // Y vector
            R_acc[0][1] = iop[3]; R_acc[1][1] = iop[4]; R_acc[2][1] = iop[5];
            // Z vector (cross product of X and Y)
            torch::Tensor x_vec = R.index({torch::indexing::Slice(), 0});
            torch::Tensor y_vec = R.index({torch::indexing::Slice(), 1});
            torch::Tensor z_vec = torch::cross(x_vec, y_vec, /*dim=*/ 0);
            
            R.index_put_({torch::indexing::Slice(), 2}, z_vec);

            torch::Tensor T = torch::tensor({ipp[0], ipp[1], ipp[2]}, torch::kFloat32);
            
            // Calculate offset from top-left corner to image center in world coordinates
            // Offset = (W/2 * ColSpacing * ColVector) + (H/2 * RowSpacing * RowVector)
            torch::Tensor offset_to_center = torch::zeros({3}, torch::kFloat32);
            for (int k = 0; k < 3; ++k)
            {
                // Note the correct pairing: H/2 with RowVector (iop[k]) and W/2 with ColVector (iop[k+3])
                offset_to_center[k] = (static_cast<double>(W) / 2.0 * ps[1] * iop[k+3]) + 
                                      (static_cast<double>(H) / 2.0 * ps[0] * iop[k]);
            }
            
            T += offset_to_center;
            torch::Tensor pose = torch::eye(4, torch::kFloat32);
            pose.index_put_({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}, R);
            pose.index_put_({torch::indexing::Slice(0, 3), 3}, T);
            
            // Convert to OpenGL camera coordinates
            // This negates the local Y and Z axes of the camera.
            pose.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1,3)}, pose.index({torch::indexing::Slice(), torch::indexing::Slice(1,3)}) * -1.0f);
            
            // --- 5. Process Pixel Data for Image Tensor and Point Cloud ---
            dcmcore::Buffer raw_pixel_buffer;
            size_t pixel_data_len = 0;
            thread_local_dataSet.GetBuffer(DicomTags_NG::PixelData, raw_pixel_buffer, pixel_data_len);

            std::vector<unsigned char> image_bytes(W * H);

            double hu_min = WC - WW / 2.0;
            double hu_max = WC + WW / 2.0;
            double window_range = WW > 0 ? WW : 1.0;

            for (unsigned short r = 0; r < H; ++r)
            {
                for (unsigned short c = 0; c < W; ++c)
                {
                    size_t pixel_idx = r * W + c;
                    int16_t raw_val_s16;
                    memcpy(&raw_val_s16, raw_pixel_buffer.data() + pixel_idx * 2, sizeof(int16_t));
                     if (thread_local_dataSet.endian() != dcmcore::PlatformEndian()) {
                        raw_val_s16 = dcmcore::byteswap(raw_val_s16);
                    }
                    double raw_value_double = static_cast<double>(raw_val_s16);

                    // Create 8-bit image
                    double norm_val = std::max(0.0, std::min(1.0, (raw_value_double - hu_min) / window_range));
                    image_bytes[pixel_idx] = static_cast<unsigned char>(norm_val * 255.0);

                    if ((r % POINT_CLOUD_DOWNSAMPLE == 0) && (c % POINT_CLOUD_DOWNSAMPLE == 0))
                    {
                        if (raw_value_double > HU_THRESHOLD)
                        {
                             std::array<double, 3> pt_3d;
                             // P = IPP + (c * ColSpacing * ColVector) + (r * RowSpacing * RowVector)
                             for(int k=0; k<3; ++k)
                             {
                                // correct pairing: r with RowVector (iop[k]) and c with ColVector (iop[k+3])
                                pt_3d[k] = ipp[k] + (c * ps[1] * iop[k + 3]) + (r * ps[0] * iop[k]);
                             }

                             local_points.push_back(pt_3d);
                             unsigned char g_byte = static_cast<unsigned char>(norm_val * 255.0);
                             local_colors.push_back({g_byte, g_byte, g_byte});
                        }
                    }
                }
            }

            // Convert image bytes to a torch::Tensor
            cam.image_tensor = torch::from_blob(image_bytes.data(), {H, W}, torch::kU8).clone();
            cam.image_tensor = cam.image_tensor.to(torch::kFloat32) / 255.0f; // Normalize
            cam.image_tensor = cam.image_tensor.unsqueeze(-1).repeat({1, 1, 3}); // [H, W] -> [H, W, 1] -> [H, W, 3]

            // Add the completed camera and its pose to the local vectors for this thread
            local_cameras.push_back(std::move(cam));
            local_poses.push_back({current_id, pose});
        }

        #pragma omp critical
        {
            cameras_accumulator.insert(cameras_accumulator.end(), std::make_move_iterator(local_cameras.begin()), std::make_move_iterator(local_cameras.end()));
            poses_accumulator.insert(poses_accumulator.end(), std::make_move_iterator(local_poses.begin()), std::make_move_iterator(local_poses.end()));
            all_points_3d.insert(all_points_3d.end(), local_points.begin(), local_points.end());
            all_colors_8bit.insert(all_colors_8bit.end(), local_colors.begin(), local_colors.end());
        }
    }

    // 1. Sort the cameras by their ID to ensure a consistent order (0, 1, 2, ...).
    std::sort(cameras_accumulator.begin(), cameras_accumulator.end(), [](const CameraYassa& a, const CameraYassa& b)
    {
        return a.id < b.id;
    });

    // 2. Sort the poses by their associated ID to match the camera order.
    std::sort(poses_accumulator.begin(), poses_accumulator.end(), [](const auto& a, const auto& b)
    {
        return a.first < b.first; // 'first' is the integer ID in the std::pair
   });
    
    // 3. Create the final list of pose tensors, which are now correctly ordered.
    std::vector<torch::Tensor> poses_list;
    poses_list.reserve(poses_accumulator.size());
    for (const auto& pose_pair : poses_accumulator)
    {
       poses_list.push_back(pose_pair.second);
   }
    
    ret.cameras = std::move(cameras_accumulator);

    //// --- 6. Scale and Center Poses and Points ---
    torch::Tensor poses_tensor = torch::stack(poses_list, 0);
    torch::Tensor R_align = get_alignment_matrix(up_direction_choice);

    // Create a 4x4 transformation matrix for poses
    torch::Tensor pose_align_matrix = torch::eye(4, torch::kFloat32);
    pose_align_matrix.index_put_({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}, R_align);

    // Apply the rotation to all camera poses
    // P_new = M_align * P_old
    poses_tensor = torch::matmul(pose_align_matrix.unsqueeze(0), poses_tensor);
    auto [scaled_poses, translation, scale] = autoScaleAndCenterPoses(poses_tensor);
    ret.translation = translation;
    ret.scale = scale;

    // Update each camera with its final, scaled pose
    for (size_t i = 0; i < ret.cameras.size(); ++i)
    {
        ret.cameras[i].camToWorld = scaled_poses[i];
    }

    // --- 7. Finalize Point Cloud ---
    if (!all_points_3d.empty())
    {
        // Convert std::vector of arrays to a flat vector for torch::from_blob
        std::vector<float> points_flat;
        points_flat.reserve(all_points_3d.size() * 3);
        for(const auto& p : all_points_3d)
        {
            points_flat.push_back(static_cast<float>(p[0]));
            points_flat.push_back(static_cast<float>(p[1]));
            points_flat.push_back(static_cast<float>(p[2]));
        }

        std::vector<unsigned char> colors_flat;
        colors_flat.reserve(all_colors_8bit.size() * 3);
        for(const auto& c : all_colors_8bit)
        {
            colors_flat.push_back(c[0]);
            colors_flat.push_back(c[1]);
            colors_flat.push_back(c[2]);
        }

        torch::Tensor points_xyz_tensor = torch::from_blob(points_flat.data(), {static_cast<long>(all_points_3d.size()), 3}, torch::kFloat32).clone();
        torch::Tensor points_rgb_tensor = torch::from_blob(colors_flat.data(), {static_cast<long>(all_colors_8bit.size()), 3}, torch::kU8).clone();

        // p_new = R_align * p_old
        points_xyz_tensor = torch::matmul(points_xyz_tensor, R_align.t());
        
        // Apply the same transformation to the point cloud
        ret.points.xyz = (points_xyz_tensor - ret.translation) * ret.scale;


        torch::Tensor min_coords_after_scale = std::get<0>(torch::min(ret.points.xyz, 0));
        float lowest_y = min_coords_after_scale[1].item<float>();
        const float desired_y_base = -1.5f;
        float y_shift_needed = desired_y_base - lowest_y;
        torch::Tensor vertical_shift = torch::tensor({0.0f, y_shift_needed, 0.0f}, ret.points.xyz.options());

        ret.points.xyz += vertical_shift;
        ret.points.rgb = points_rgb_tensor.to(torch::kFloat32) / 255.0f; // Normalize colors
    }
    
    std::cout << "DICOM processing complete. Loaded " << ret.cameras.size() << " cameras and "
              << (ret.points.xyz.defined() ? ret.points.xyz.size(0) : 0) << " 3D points." << std::endl;

    return ret;
}
