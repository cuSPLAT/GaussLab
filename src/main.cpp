#include <core/engine.h>

int main() {
    try {
        GaussLabEngine engine;
        engine.run();
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
