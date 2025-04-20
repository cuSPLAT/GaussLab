#include "dicom_reader.h"

#include <filesystem>
#include <string>
#include <algorithm>
#include <cmath>
#include <torch/types.h>
#include <vega/dicom/file.h>
#include <vega/dictionary/dictionary.h>



namespace fs = std::filesystem;

DicomReader::DicomReader(const std::string& dictionary) {
    if (dictionary == "")
        vega::dictionary::set_dictionary(DEFAULT_DICOM_DICTIONARY);
    else
        vega::dictionary::set_dictionary(dictionary);
}

static bool zAxisComparison(vega::dicom::File &f1, vega::dicom::File &f2) {
    auto f1_pos = getDataManipulator(f1, vega::dictionary::ImagePositionPatient);
    auto f2_pos = getDataManipulator(f2, vega::dictionary::ImagePositionPatient);
    return static_cast<float>(f1_pos->begin()[2]) < static_cast<float>(f2_pos->begin()[2]);
}

Geometry DicomReader::readDirectory(const std::string& path) {
    for (const auto &entry: fs::directory_iterator(path)) {
        dicomFiles.emplace_back(entry.path().string());
    }
    std::sort(dicomFiles.begin(), dicomFiles.end(), zAxisComparison);

    auto bits = *getDataManipulator(dicomFiles[0], vega::dictionary::BitsAllocated)->begin();
    auto width = *getDataManipulator(dicomFiles[0], vega::dictionary::Columns)->begin();
    auto height = *getDataManipulator(dicomFiles[0], vega::dictionary::Rows)->begin();
    const uint32_t length = width * height * dicomFiles.size();
    float* volume = new float[length];

    if (bits == 16) {
        size_t index = 0;
        for (const auto& image : dicomFiles) {
            auto pixel_data = getDataManipulator(image, vega::dictionary::PixelData_OW);
            auto slope = static_cast<float>(*getDataManipulator(image, vega::dictionary::RescaleSlope)->begin());
            auto intercept = static_cast<float>(*getDataManipulator(image, vega::dictionary::RescaleIntercept)->begin());

            for (auto it = pixel_data->begin(); it != pixel_data->end(); ++it) {
                volume[index++] = it->u * slope + intercept;
            }
        }

        torch::Tensor volume_tensor = torch::from_blob(volume, {static_cast<long>(length)}, torch::kFloat);
        volume_tensor = (volume_tensor - volume_tensor.min()) / (volume_tensor.max() - volume_tensor.min());
    }

    Geometry geom;
    geom.nVoxelX = width;
    geom.nVoxelY = height;
    geom.nVoxelZ = dicomFiles.size();

    auto spacing = getDataManipulator(dicomFiles[0], vega::dictionary::PixelSpacing);
    geom.sVoxelX = static_cast<float>(spacing->begin()[0]);
    geom.sVoxelY = static_cast<float>(spacing->begin()[1]);
    geom.sVoxelZ = static_cast<float>(*getDataManipulator(dicomFiles[0], vega::dictionary::SliceThickness)->begin());

    geom.offOrigX = new float(static_cast<float>(getDataManipulator(dicomFiles[0], vega::dictionary::ImagePositionPatient)->begin()[0]));
    geom.offOrigY = new float(static_cast<float>(getDataManipulator(dicomFiles[0], vega::dictionary::ImagePositionPatient)->begin()[1]));
    geom.offOrigZ = new float(static_cast<float>(getDataManipulator(dicomFiles[0], vega::dictionary::ImagePositionPatient)->begin()[2]));

    geom.nDetecU = geom.nVoxelX;
    geom.nDetecV = geom.nVoxelY;
    geom.sDetecU = geom.sVoxelX;
    geom.sDetecV = geom.sVoxelY;
    geom.dDetecU = 1.0f;
    geom.dDetecV = 1.0f;
    geom.offDetecU = new float(0.0f);
    geom.offDetecV = new float(0.0f);

    geom.DSO = new float(1000.0f);
    geom.DSD = new float(1500.0f);

    geom.dRoll = new float(0.0f);
    geom.dPitch = new float(0.0f);
    geom.dYaw = new float(0.0f);

    geom.COR = new float(0.0f);
    geom.unitX = geom.sVoxelX;
    geom.unitY = geom.sVoxelY;
    geom.unitZ = geom.sVoxelZ;

    geom.maxLength = std::sqrt(
        (geom.nVoxelX * geom.sVoxelX) * (geom.nVoxelX * geom.sVoxelX) +
        (geom.nVoxelY * geom.sVoxelY) * (geom.nVoxelY * geom.sVoxelY) +
        (geom.nVoxelZ * geom.sVoxelZ) * (geom.nVoxelZ * geom.sVoxelZ)
    );

    geom.accuracy = 0.01f;

    auto orient = getDataManipulator(dicomFiles[0], vega::dictionary::ImageOrientationPatient);
    
    // std::cout << "Orientation: ";
    // for (size_t i = 0; i < orient->size(); ++i) {
    //     std::cout << orient->begin()[i] << " ";
    // }
    std::cout << std::endl;

    float row_x = orient->begin()[0];
    float row_y = orient->begin()[1];
    float row_z = orient->begin()[2];
    float col_x = orient->begin()[3];
    float col_y = orient->begin()[4];
    float col_z = orient->begin()[5];


    float norm_x = row_y * col_z - row_z * col_y;
    float norm_y = row_z * col_x - row_x * col_z;
    float norm_z = row_x * col_y - row_y * col_x;

    
    geom.alpha = col_x ;                    
    geom.theta = col_y;
    geom.psi   = col_z;

    // // ---------------- Print ----------------
    // std::cout << "Geometry:" << std::endl;
    // std::cout << "  nVoxelX: " << geom.nVoxelX << std::endl;
    // std::cout << "  nVoxelY: " << geom.nVoxelY << std::endl;
    // std::cout << "  nVoxelZ: " << geom.nVoxelZ << std::endl;
    // std::cout << "  sVoxelX: " << geom.sVoxelX << std::endl;
    // std::cout << "  sVoxelY: " << geom.sVoxelY << std::endl;
    // std::cout << "  sVoxelZ: " << geom.sVoxelZ << std::endl;
    // std::cout << "  offOrigX: " << *geom.offOrigX << std::endl;
    // std::cout << "  offOrigY: " << *geom.offOrigY << std::endl;
    // std::cout << "  offOrigZ: " << *geom.offOrigZ << std::endl;
    // std::cout << "  nDetecU: " << geom.nDetecU << std::endl;
    // std::cout << "  nDetecV: " << geom.nDetecV << std::endl;
    // std::cout << "  sDetecU: " << geom.sDetecU << std::endl;
    // std::cout << "  sDetecV: " << geom.sDetecV << std::endl;
    // std::cout << "  dDetecU: " << geom.dDetecU << std::endl;
    // std::cout << "  dDetecV: " << geom.dDetecV << std::endl;
    // std::cout << "  offDetecU: " << *geom.offDetecU << std::endl;
    // std::cout << "  offDetecV: " << *geom.offDetecV << std::endl;
    // std::cout << "  DSO: " << *geom.DSO << std::endl;
    // std::cout << "  DSD: " << *geom.DSD << std::endl;
    // std::cout << "  dRoll: " << *geom.dRoll << std::endl;
    // std::cout << "  dPitch: " << *geom.dPitch << std::endl;
    // std::cout << "  dYaw: " << *geom.dYaw << std::endl;
    // std::cout << "  alpha: " << geom.alpha << std::endl;
    // std::cout << "  theta: " << geom.theta << std::endl;
    // std::cout << "  psi: " << geom.psi << std::endl;
    // std::cout << "  COR: " << *geom.COR << std::endl;
    // std::cout << "  maxLength: " << geom.maxLength << std::endl;
    // std::cout << "  accuracy: " << geom.accuracy << std::endl;
    // std::cout << "  unitX: " << geom.unitX << std::endl;
    // std::cout << "  unitY: " << geom.unitY << std::endl;
    // std::cout << "  unitZ: " << geom.unitZ << std::endl;

    delete[] volume;
    return geom;
}

// int main(){
//     DicomReader dicomReader;
//     Geometry geom = dicomReader.readDirectory("/home/zain/Downloads/ffe79971-17a080e0-0092e608-f54c9108-89d4cb85/74640d444c104e879aa62d2949bd98b8 Anonymized479/Unknown Study/CT");
//     // Use geom as needed
//     return 0;
// }
