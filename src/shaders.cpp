#include "shaders.h"

// defining the shaders in variables just to prevent the
// headache of bundling the shader files with the binary
const char* Shaders::vertexShader = R"(
    #version 460 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 inColor;

    out vec4 vertexColor;

    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        gl_Position = projection * view * vec4(aPos, 1.0);
        gl_PointSize = 1.5f;
        vertexColor = vec4(inColor * 0.282 + 0.5, 1.0);
    }
)";

const char* Shaders::fragmentShader = R"(
    #version 460 core
    in vec4 vertexColor;

    out vec4 FragColor;

    void main() {
        FragColor = vertexColor;
    }
)";

const char* Shaders::viewMatMulCompute = R"(
    #version 460 core
    layout (location = 0) in vec3 aPos;
    layout (std430, binding = 0) buffer gaussianDepth {
        float z_depth[];
    };

    layout (std430, binding = 1) buffer gaussianIndices {
        float gaussian_indices[];
    };

    uniform mat4 view;

    void main() {
        vec4 world_space = view * vec4(aPos, 1.0);
        z_depth[gl_VertexID] = world_space.z;
        gaussian_indices[gl_VertexID] = gl_VertexID;
    }
)";

const char* Shaders::gaussianVertexShader = R"(
    #version 460 core

    void main() {

    }
)";
