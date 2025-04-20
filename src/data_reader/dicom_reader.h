#include <atomic>
#include <string>
#include <vega/dicom/file.h>

#define DEFAULT_DICOM_DICTIONARY "/home/Abdelrahman/projects/gradproject/GausStudio/dictionary.txt"

#define getDataManipulator(file, tag) file.data_set()->element<tag>()->manipulator()




struct Geometry {
    int   nVoxelX, nVoxelY, nVoxelZ;
    float sVoxelX, sVoxelY, sVoxelZ;
    float dVoxelX, dVoxelY, dVoxelZ;
    float *offOrigX, *offOrigY, *offOrigZ;
    float* DSO;
    int   nDetecU, nDetecV;
    float sDetecU, sDetecV;
    float dDetecU, dDetecV;
    float *offDetecU, *offDetecV;
    float* DSD;
    float* dRoll;
    float* dPitch;
    float* dYaw;
    float unitX, unitY, unitZ;
    float alpha, theta, psi;
    float* COR;
    float maxLength;
    float accuracy;
};

class DicomReader {
    std::vector<vega::dicom::File> dicomFiles;

public:
    DicomReader(const std::string& dictionary = "");
    
    Geometry readDirectory(const std::string& path);
};
