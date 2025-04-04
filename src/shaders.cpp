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
    layout (location = 2) in vec2 quadPosition;
    layout (std430, binding = 3) buffer sortedIndices {
        int indices[];
    };
    layout (std430, binding = 4) buffer gaussianBuffer {
        float gaussianData[];
    };

    #define POS_OFFSET 0
    #define SH_OFFSET 3
    #define OPACITY_OFFSET 6
    #define SCALE_OFFSET 7
    #define ROT_OFFSET 10
    
    #define STRIDE 14

    uniform mat4 view;
    uniform mat4 projection;
    uniform vec3 hfov_focal;

    out vec3 outColor;
    out float opacity;

    vec3 get_vec3(int offset) {
        return vec3(gaussianData[offset], gaussianData[offset + 1], gaussianData[offset + 2]);
    }

    vec4 get_vec4(int offset) {
        return vec4(gaussianData[offset], gaussianData[offset + 1], gaussianData[offset + 2], gaussianData[offset+3]);
    }

    // I think this can be delegated to the CPU, and be done once before a render
    // too much overhead
    mat3 computeCov3D(vec4 rots, vec3 scales) {
      float scaleMod = 1.0f;
      vec3 firstRow = vec3(
        1.f - 2.f * (rots.z * rots.z + rots.w * rots.w),
        2.f * (rots.y * rots.z - rots.x * rots.w),      
        2.f * (rots.y * rots.w + rots.x * rots.z)       
      );
      vec3 secondRow = vec3(
        2.f * (rots.y * rots.z + rots.x * rots.w),       
        1.f - 2.f * (rots.y * rots.y + rots.w * rots.w), 
        2.f * (rots.z * rots.w - rots.x * rots.y)        
      );
      vec3 thirdRow = vec3(
        2.f * (rots.y * rots.w - rots.x * rots.z),       
        2.f * (rots.z * rots.w + rots.x * rots.y),     
        1.f - 2.f * (rots.y * rots.y + rots.z * rots.z) 
      );

      mat3 scaleMatrix = mat3(
        scaleMod * scales.x, 0, 0, 
        0, scaleMod * scales.y, 0,
        0, 0, scaleMod * scales.z
      );

      mat3 rotMatrix = mat3(
        firstRow,
        secondRow,
        thirdRow
      );
      mat3 mMatrix = scaleMatrix * rotMatrix;
      mat3 sigma = transpose(mMatrix) * mMatrix;
      return sigma;
    };


    void main() {
        int index = indices[gl_InstanceID];
        int start_index = index * STRIDE;
        
        vec3 mean = get_vec3(start_index);
        vec3 base_SH = get_vec3(start_index + SH_OFFSET);
        vec3 scale = get_vec3(start_index + SCALE_OFFSET);
        vec4 rotation = get_vec4(start_index + ROT_OFFSET);

        mat3 cov3D = computeCov3D(rotation, scale);

        vec4 view_space = view * vec4(mean, 1);
        vec4 pos2d = projection * view_space;
        // Apply perspective divison here because the 2d coordinates will be 
        // used in other calculations
        pos2d.xyz = pos2d.xyz / pos2d.w;
        pos2d.w = 1.0f;

        vec2 width_height = 2 * hfov_focal.xy * hfov_focal.z;

        opacity = gaussianData[start_index + OPACITY_OFFSET];
    }
)";

const char* Shaders::gaussianFragmentShader = R"(
    #version 460 core

    in vec3 outColor;
    in float opacity;

    out vec4 FragColor;

    void main() {

    }
)";
