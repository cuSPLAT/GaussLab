#include "dicom_reader.h"

#include <ATen/ops/from_blob.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <cuda.h>
#include <cuda_runtime.h>

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

    auto pixel_representation = getDataManipulator(dicomFiles[0], vega::dictionary::PixelRepresentation);
    auto signed_value = *pixel_representation->begin();
    
    // allocate a 1D block that will be usesd by Tigre to do the projections
    const u_int32_t length =  width * height * dicomFiles.size();
    const int slice_area = width * height;
    std::unique_ptr<float[]> volume = std::make_unique<float[]>(length);

    TIME_SANDWICH_START(load);
    if (bits == 16) {
        loadDataMultithreaded(volume.get(), signed_value, slice_area);
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

void DicomReader::loadDataMultithreaded(float* volume, int signed_value, int slice_area) {
    // 2 threads are harcoded that can give good performance, maybe
    // later this can be changed to be chosen
    std::thread t1(
        &DicomReader::loadData_thread, this, 0,
        volume, signed_value, slice_area
    );
    std::thread t2(
        &DicomReader::loadData_thread, this, 1,
        volume, signed_value, slice_area
    );
    
    t1.join();
    t2.join();
}


void DicomReader::loadData_thread(int threadidx, float* buffer, int signed_value, int slice_area) {
    const int stride = dicomFiles.size() / 2 + 1;
    const int start_idx = stride * threadidx;

    for (size_t i = start_idx; i < start_idx + stride && i < dicomFiles.size(); i++) {
        auto pixel_data = getDataManipulator(dicomFiles[i], vega::dictionary::PixelData_OW);
        auto slope_manipulator = getDataManipulator(dicomFiles[i], vega::dictionary::RescaleSlope);
        auto intercept_manipulator = getDataManipulator(dicomFiles[i], vega::dictionary::RescaleIntercept);

        float slope = static_cast<float>(*slope_manipulator->begin());
        float intercept = static_cast<float>(*intercept_manipulator->begin());

        torch::Tensor slice_tensor;
        if (signed_value) {
            int16_t* pixel_array = reinterpret_cast<int16_t*>(&(*pixel_data->begin())); // WTF ?
            slice_tensor = torch::from_blob(pixel_array, {slice_area}, torch::kInt16).to(torch::kFloat);
        } else {
            uint16_t* pixel_array = reinterpret_cast<uint16_t*>(&(*pixel_data->begin()));
            slice_tensor = torch::from_blob(pixel_array, {slice_area}, torch::kUInt16).to(torch::kFloat);
        }

        slice_tensor = slice_tensor * slope + intercept;
        const int offset = i * slice_area;
        std::memcpy(buffer + offset, slice_tensor.const_data_ptr(), slice_area * sizeof(float));
        loadingProgress++;
    }
}

void DicomReader::cleanupThreads() {
    LOG("Cleaning running threads for DICOM Loading")

    for (auto& thread : threads)
        thread.join();
}
