#include "dicom_reader.h"

#include <filesystem>

#include <string>

#include <torch/types.h>
#include <tuple>

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

std::tuple<float*, int, int, int> DicomReader::readDirectory(const std::string& path) {
    for (const auto &entry: fs::directory_iterator(path)) {
        dicomFiles.emplace_back(entry.path().string());
    }
    std::sort(dicomFiles.begin(), dicomFiles.end(), zAxisComparison);
    
    // How many bits are allocated for a single pixel
    auto manipulator = getDataManipulator(dicomFiles[0], vega::dictionary::BitsAllocated);
    unsigned short bits = *manipulator->begin();

    auto width_manipulator = getDataManipulator(dicomFiles[0], vega::dictionary::Columns);
    auto height_manipulator = getDataManipulator(dicomFiles[0], vega::dictionary::Rows);
    unsigned short width = *width_manipulator->begin(), height = *height_manipulator->begin();
    
    // allocate a 1D block that will be usesd by Tigre to do the projections
    const u_int32_t length =  width * height * dicomFiles.size();
    float* volume = new float[length];

    if (bits == 16) {
        size_t index = 0;
        for (const auto& image : dicomFiles) {
            auto pixel_data = getDataManipulator(image, vega::dictionary::PixelData_OW);
            auto slope_manipulator = getDataManipulator(image, vega::dictionary::RescaleSlope);
            auto intercept_manipulator = getDataManipulator(image, vega::dictionary::RescaleIntercept);

            float slope = static_cast<float>(*slope_manipulator->begin());
            float intercept = static_cast<float>(*intercept_manipulator->begin());

            for (auto it = pixel_data->begin(); it != pixel_data->end(); it++) {
                volume[index] = it->u * slope + intercept;
                index++;
            }
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

    return std::make_tuple(volume, width, height, dicomFiles.size());
}
