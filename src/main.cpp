#include <core/engine.h>

int main() {
    try {
        GaussLabEngine engine;
        engine.run();
    } catch (std::exception& e) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
