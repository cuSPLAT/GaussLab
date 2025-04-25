#include <atomic>
#define DEFAULT_DICOM_DICTIONARY "/home/Abdelrahman/projects/gradproject/GausStudio/dictionary.txt"

#define getDataManipulator(file, tag) file.data_set()->element<tag>()->manipulator()

#include <string>
#include <thread>

#include <vega/dicom/file.h>


class DicomReader {
    std::vector<vega::dicom::File> dicomFiles;

    std::vector<std::thread> threads;

public:
    std::atomic<std::uint64_t> loadingProgress {0};
    // initialized with a large number so the loading bar can work instantly
    std::atomic<std::uint64_t> totalSize {10000000};

    struct DicomData {
        std::unique_ptr<float[]> buffer;
        int width, length, height;

        std::atomic_flag readable {};
    };

    DicomData loadedData {};

    DicomReader(const std::string& dictionary = "");
    
    void readDirectory(const std::string& path);

    void launchReaderThread(const std::string& path);
    void cleanupThreads();
};
