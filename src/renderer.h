#include <glad/glad.h>

class Renderer {
    GLuint VBO, VAO;
    GLuint frameBuffer;
    GLuint shaderProgram;

    GLuint rendererBuffer;

    unsigned int width, height;

public:
    Renderer(int width, int height);
    void generateInitialBuffers();
    void initializeRendererBuffer();

    GLuint getRenderBuffer();

    void render();
};
