#include "dicom_reader.h"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

#include <torch/types.h>

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
    auto f1_position_manipulator = getDataManipulator(f1, vega::dictionary::ImagePositionPatient);
    auto f2_position_manipulator = getDataManipulator(f2, vega::dictionary::ImagePositionPatient);
    
    return static_cast<float>(f1_position_manipulator->begin()[2]) 
        > static_cast<float>(f2_position_manipulator->begin()[2]);
}

void DicomReader::readDirectory(const std::string& path) {
    for (const auto &entry: fs::directory_iterator(path)) {
        dicomFiles.emplace_back(entry.path().string());
    }
    totalSize = dicomFiles.size();
    std::sort(dicomFiles.begin(), dicomFiles.end(), zAxisComparison);
    
    // How many bits are allocated for a single pixel
    auto manipulator = getDataManipulator(dicomFiles[0], vega::dictionary::BitsAllocated);
    unsigned short bits = *manipulator->begin();

    auto width_manipulator = getDataManipulator(dicomFiles[0], vega::dictionary::Columns);
    auto height_manipulator = getDataManipulator(dicomFiles[0], vega::dictionary::Rows);
    unsigned short width = *width_manipulator->begin(), height = *height_manipulator->begin();
    
    // allocate a 1D block that will be usesd by Tigre to do the projections
    const u_int32_t length =  width * height * dicomFiles.size();
    const int single_slice = width * height;
    std::unique_ptr<float[]> volume = std::make_unique<float[]>(length);

    TIME_SANDWICH_START(load);
    if (bits == 16) {
        size_t index = 0;
        for (size_t i = 0; i < dicomFiles.size(); i++) {
            auto pixel_data = getDataManipulator(dicomFiles[i], vega::dictionary::PixelData_OW);
            auto slope_manipulator = getDataManipulator(dicomFiles[i], vega::dictionary::RescaleSlope);
            auto intercept_manipulator = getDataManipulator(dicomFiles[i], vega::dictionary::RescaleIntercept);

            float slope = static_cast<float>(*slope_manipulator->begin());
            float intercept = static_cast<float>(*intercept_manipulator->begin());

            uint16_t* pixel_array = reinterpret_cast<uint16_t*>(pixel_data->raw_value()->data());
            torch::Tensor slice_tensor = torch::from_blob(pixel_array, {single_slice}, torch::kUInt16).to(torch::kFloat);
            slice_tensor = slice_tensor * slope + intercept;

            const int offset = i * single_slice;
            std::memcpy(volume.get() + offset, slice_tensor.const_data_ptr(), single_slice * sizeof(float));
            loadingProgress++;
        }

        // I just created this tensor to use the optimizations that torch gives me
        // and there is no overhead because torch does not clone the data
        //torch::Tensor volume_tensor = torch::from_blob(volume, {length}, torch::kFloat);
        //auto max_val = volume_tensor.max(), min_val = volume_tensor.min();
        //volume_tensor -= min_val;
        //volume_tensor /= max_val - min_val;

    } else {
        //TODO: implement for DICOM files that have less that 2 bytes for 
        // a single image
    }
    TIME_SANDWICH_END(load)

    loadedData.buffer = std::move(volume);
    loadedData.width = width;
    loadedData.length = height;
    loadedData.height = dicomFiles.size();

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

void DicomReader::cleanupThreads() {
    LOG("Cleaning running threads for DICOM Loading")

    for (auto& thread : threads)
        thread.join();
}
