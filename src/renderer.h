#include <glad/glad.h>

class Renderer {
    GLuint VBO, VAO;
    GLuint frameBuffer;
    GLuint shaderProgram;

public:
    Renderer();
    void generateInitialBuffers();

    void render();
};
