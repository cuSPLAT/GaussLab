#include "dicom_reader.h"

#include <ATen/ops/from_blob.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>
#include <algorithm>

#include <string>

#include <torch/types.h>
#include <vega/dictionary_data.h>

#include "../cuda/debug_utils.h"
#include "../debug_utils.h"

namespace fs = std::filesystem;

DicomReader::DicomReader(const std::string& dictionary) {
    // the dictionary is a file that defines each tag in the DICOM file
    // we will provide a default one but one can be defined by the user
    if (dictionary == "")
        vega::dictionary::set_dictionary(DEFAULT_DICOM_DICTIONARY);
    else vega::dictionary::set_dictionary(dictionary);
}

static bool zAxisComparison(vega::dicom::File &f1, vega::dicom::File &f2) {
    try {
        auto f1_position_manipulator = getDataManipulator(f1, vega::dictionary::ImagePositionPatient);
        auto f2_position_manipulator = getDataManipulator(f2, vega::dictionary::ImagePositionPatient);
        
        if (!f1_position_manipulator || !f2_position_manipulator) {
            // If either file doesn't have position data, maintain original order
            return false;
        }
        
        if (f1_position_manipulator->size() < 3 || f2_position_manipulator->size() < 3) {
            // If position data is incomplete, maintain original order
            return false;
        }
        
        return static_cast<float>(f1_position_manipulator->begin()[2]) 
            > static_cast<float>(f2_position_manipulator->begin()[2]);
    } catch (const std::exception& e) {
        // Return false to maintain original order on error
        return false;
    }
}

void DicomReader::readDirectory(const std::string& path) {
    
    // Clear previous data
    dicomFiles.clear();
    loadingProgress = 0;
    
    for (const auto &entry: fs::directory_iterator(path)) {
        // Only process files, not directories
        if (entry.is_regular_file()) {
            try {
                dicomFiles.emplace_back(entry.path().string());
                if (dicomFilePath.empty()) {
                    dicomFilePath = entry.path().string();
                }
            } catch (const std::exception& e) {
            }
        }
    }
    
    totalSize = dicomFiles.size();
    
    if (dicomFiles.empty()) {
        return;
    }
    
    // Validate that we have at least one valid DICOM file
    if (dicomFiles.size() == 0) {
        return;
    }
    
    try {
        std::sort(dicomFiles.begin(), dicomFiles.end(), zAxisComparison);
    } catch (const std::exception& e) {
        // Continue without sorting if it fails
    }
    
    // How many bits are allocated for a single pixel
    auto manipulator = getDataManipulator(dicomFiles[0], vega::dictionary::BitsAllocated);
    if (!manipulator) {
        return;
    }
    unsigned short bits = *manipulator->begin();
    
    auto width_manipulator = getDataManipulator(dicomFiles[0], vega::dictionary::Columns);
    auto height_manipulator = getDataManipulator(dicomFiles[0], vega::dictionary::Rows);
    
    if (!width_manipulator || !height_manipulator) {
        return;
    }
    
    unsigned short width = *width_manipulator->begin(), height = *height_manipulator->begin();
    
    // Validate dimensions
    if (width == 0 || height == 0) {
        return;
    }
    
    auto pixel_representation = getDataManipulator(dicomFiles[0], vega::dictionary::PixelRepresentation);
    if (!pixel_representation) {
        return;
    }
    auto signed_value = *pixel_representation->begin();
    
    // allocate a 1D block that will be used by Tigre to do the projections
    const u_int32_t length = width * height * dicomFiles.size();
    const int slice_area = width * height;
    
    // Check for potential integer overflow
    if (length / width / height != dicomFiles.size()) {
        return;
    }
    
    
    try {
        std::unique_ptr<float[]> volume = std::make_unique<float[]>(length);
        TIME_SANDWICH_START(load);
        if (bits == 16) {
            loadDataMultithreaded(volume.get(), signed_value, slice_area);
        } else {
            //TODO: implement for DICOM files that have less that 2 bytes for 
            // a single image
            return;
        }
        TIME_SANDWICH_END(load)
        loadedData.buffer = std::move(volume);
        loadedData.width = width;
        loadedData.length = height;
        loadedData.height = dicomFiles.size();
    } catch (const std::bad_alloc& e) {
        return;
    } catch (const std::exception& e) {
        return;
    }
    
    // Extract pixel spacing (X, Y) and slice thickness (Z)
    try {
        auto spacing_manip = getDataManipulator(dicomFiles[0], vega::dictionary::PixelSpacing);
        if (spacing_manip && spacing_manip->size() >= 2) {
            loadedData.pixelSpacingX = static_cast<float>(spacing_manip->at(0));
            loadedData.pixelSpacingY = static_cast<float>(spacing_manip->at(1));
        } else {
            loadedData.pixelSpacingX = 1.0f;
            loadedData.pixelSpacingY = 1.0f;
        }
    } catch (...) {
        loadedData.pixelSpacingX = 1.0f;
        loadedData.pixelSpacingY = 1.0f;
    }
    
    try {
        auto thick_manip = getDataManipulator(dicomFiles[0], vega::dictionary::SliceThickness);
        if (thick_manip && thick_manip->size() >= 1) {
            loadedData.sliceThickness = static_cast<float>(thick_manip->at(0));
        } else {
            loadedData.sliceThickness = 1.0f;
        }
    } catch (...) {
        loadedData.sliceThickness = 1.0f;
    }
    
    // maybe later it is better to notify a thread
    loadedData.readable.test_and_set();
}

// A wrapper for starting the reader from another thread
void DicomReader::launchReaderThread(const std::string& path) {
    std::thread thread([path, this]() {
        readDirectory(path);
    });
    threads.push_back(std::move(thread));
}

void DicomReader::loadDataMultithreaded(float* volume, int signed_value, int slice_area) {
    // Use std::thread::hardware_concurrency() to get optimal thread count
    // but cap it to avoid too many threads for small datasets
    const unsigned int max_threads = std::thread::hardware_concurrency();
    const unsigned int num_files = dicomFiles.size();
    const unsigned int num_threads = std::min(max_threads, std::min(num_files, 4u)); // Cap at 4 threads max
    
    
    if (num_threads == 1) {
        // Single thread case
        loadData_thread(0, volume, signed_value, slice_area);
        return;
    }
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.emplace_back(
            &DicomReader::loadData_thread, this, i,
            volume, signed_value, slice_area
        );
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

void DicomReader::loadData_thread(int threadidx, float* buffer, int signed_value, int slice_area) {
    const unsigned int num_files = dicomFiles.size();
    const unsigned int max_threads = std::thread::hardware_concurrency();
    const unsigned int num_threads = std::min(max_threads, std::min(num_files, 4u)); // Cap at 4 threads max
    
    const unsigned int files_per_thread = num_files / num_threads;
    const unsigned int remainder = num_files % num_threads;
    
    // Calculate start and end indices for this thread
    unsigned int start_idx = static_cast<unsigned int>(threadidx) * files_per_thread + std::min(static_cast<unsigned int>(threadidx), remainder);
    unsigned int end_idx = start_idx + files_per_thread + (static_cast<unsigned int>(threadidx) < remainder ? 1 : 0);
    

    for (unsigned int i = start_idx; i < end_idx && i < num_files; i++) {
        try {
            auto pixel_data = getDataManipulator(dicomFiles[i], vega::dictionary::PixelData_OW);
            auto slope_manipulator = getDataManipulator(dicomFiles[i], vega::dictionary::RescaleSlope);
            auto intercept_manipulator = getDataManipulator(dicomFiles[i], vega::dictionary::RescaleIntercept);

            if (!pixel_data || !slope_manipulator || !intercept_manipulator) {
                continue;
            }

            float slope = static_cast<float>(*slope_manipulator->begin());
            float intercept = static_cast<float>(*intercept_manipulator->begin());

            torch::Tensor slice_tensor;
            if (signed_value) {
                int16_t* pixel_array = reinterpret_cast<int16_t*>(&(*pixel_data->begin()));
                slice_tensor = torch::from_blob(pixel_array, {slice_area}, torch::kInt16).to(torch::kFloat);
            } else {
                uint16_t* pixel_array = reinterpret_cast<uint16_t*>(&(*pixel_data->begin()));
                slice_tensor = torch::from_blob(pixel_array, {slice_area}, torch::kUInt16).to(torch::kFloat);
            }

            slice_tensor = slice_tensor * slope + intercept;
            const int offset = i * slice_area;
            
            // Bounds check before memcpy
            if (offset + slice_area <= num_files * slice_area) {
                std::memcpy(buffer + offset, slice_tensor.const_data_ptr(), slice_area * sizeof(float));
            } else {
            }
            
            loadingProgress++;
        } catch (const std::exception& e) {
        } catch (...) {
        }
    }
}

void DicomReader::cleanupThreads() {

    for (auto& thread : threads)
        thread.join();
}
