#define DEFAULT_DICOM_DICTIONARY "/home/Abdelrahman/projects/gradproject/GausStudio/dictionary.txt"

#define getDataManipulator(file, tag) file.data_set()->element<tag>()->manipulator()

#include <string>

#include <vega/dicom/file.h>

class DicomReader {
    std::vector<vega::dicom::File> dicomFiles;

public:
    DicomReader(const std::string& dictionary = "");
    
    void readDirectory(const std::string& path);
};
