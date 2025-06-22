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
    dicomFiles.clear();
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
    
    if (dicomFiles.empty()) {
        return;
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
    
    if (length / width / height != dicomFiles.size()) {
        return;
    }
    
    std::unique_ptr<float[]> volume = std::make_unique<float[]>(length);

    //TIME_SANDWICH_START(load);
    if (bits == 16) {
        loadDataMultithreaded(volume.get(), signed_value, slice_area);
    } else {
        //TODO: implement for DICOM files that have less that 2 bytes for 
        // a single image
    }
    //TIME_SANDWICH_END(load)

    loadedData.buffer = std::move(volume);
    loadedData.width = width;
    loadedData.length = height;
    loadedData.height = dicomFiles.size();

    // Extract windowing info
    try {
        auto wc_manip = getDataManipulator(dicomFiles[0], vega::dictionary::WindowCenter);
        if (wc_manip && wc_manip->size() >= 1) {
            loadedData.windowCenter = static_cast<float>(wc_manip->at(0));
        }
    } catch (...) {}
    
    try {
        auto ww_manip = getDataManipulator(dicomFiles[0], vega::dictionary::WindowWidth);
        if (ww_manip && ww_manip->size() >= 1) {
            loadedData.windowWidth = static_cast<float>(ww_manip->at(0));
        }
    } catch (...) {}

    // Calculate min and max for sliders and default windowing
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();
    for (int i = 0; i < length; ++i) {
        if (loadedData.buffer[i] < minVal) minVal = loadedData.buffer[i];
        if (loadedData.buffer[i] > maxVal) maxVal = loadedData.buffer[i];
    }
    loadedData.dataMin = minVal;
    loadedData.dataMax = maxVal;

    if (loadedData.windowWidth == 0.0f) {
        loadedData.windowCenter = (maxVal + minVal) / 2.0f;
        loadedData.windowWidth = maxVal - minVal;
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

    // Extract other metadata
    auto dataset = dicomFiles[0].data_set();
    for (const auto& element : *dataset) {
        if (!element) continue;
        vega::Tag tag = element->tag();

        if (tag.group() == 0x0010 && tag.element() == 0x0010) loadedData.patientName = element->str();
        else if (tag.group() == 0x0008 && tag.element() == 0x0022) loadedData.scanDate = element->str();
        else if (tag.group() == 0x0008 && tag.element() == 0x0020 && loadedData.scanDate.empty()) loadedData.scanDate = element->str();
        else if (tag.group() == 0x0018 && tag.element() == 0x0015) loadedData.bodyPartExamined = element->str();
        else if (tag.group() == 0x0018 && tag.element() == 0x0010) loadedData.contrastAgent = element->str();
        else if (tag.group() == 0x0032 && tag.element() == 0x1030) loadedData.reasonForStudy = element->str();
        else if (tag.group() == 0x0032 && tag.element() == 0x1060) loadedData.requestedProcedureDescription = element->str();
        else if (tag.group() == 0x0040 && tag.element() == 0x1002) loadedData.reasonForRequestedProcedure = element->str();
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
            std::memcpy(buffer + offset, slice_tensor.const_data_ptr(), slice_area * sizeof(float));
            
            loadingProgress++;
    }
}

void DicomReader::cleanupThreads() {
    LOG_CLIENT("Cleaning running threads for DICOM Loading")

    for (auto& thread : threads)
        thread.join();
}
