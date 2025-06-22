#include <atomic>
#define DEFAULT_DICOM_DICTIONARY "/home/zain/GausStudio-main/dictionary.txt"

#define getDataManipulator(file, tag) file.data_set()->element<tag>()->manipulator()

#include <string>
#include <thread>

#include <vega/dicom/file.h>


class DicomReader {
    std::vector<vega::dicom::File> dicomFiles;
    std::string dicomFilePath="";

    std::vector<std::thread> threads;

public:
    std::atomic<std::uint64_t> loadingProgress {0};
    // initialized with a large number so the loading bar can work instantly
    std::atomic<std::uint64_t> totalSize {10000000};

    const std::vector<vega::dicom::File>& getDicomFiles() const { return dicomFiles; }
    std::string getDicomFilePaths() const { 
        return dicomFilePath; }

    struct DicomData {
        std::unique_ptr<float[]> buffer;
        int width, length, height;
        float pixelSpacingX = 1.0f;
        float pixelSpacingY = 1.0f;
        float sliceThickness = 1.0f;

        std::atomic_flag readable {};
    };

    DicomData loadedData {};

    DicomReader(const std::string& dictionary = "");
    
    void readDirectory(const std::string& path);

    void loadData_thread(int threadidx, float* buffer, int signed_value, int slice_area);
    void loadDataMultithreaded(float* volume, int signed_value, int slice_area);

    void launchReaderThread(const std::string& path);
    void cleanupThreads();
};
