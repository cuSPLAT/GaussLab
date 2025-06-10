#include "shaders.h"

// defining the shaders in variables just to prevent the
// headache of bundling the shader files with the binary
//
// This is marching cubes shader for now
const char* Shaders::vertexShader = R"(
    #version 460 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;

    out vec3 Normal;
    out vec3 FragPos;
    out vec3 vertexColor;
    out vec3 WorldPos;
    
    uniform mat4 view;
    uniform mat4 model;
    uniform mat4 projection;

    void main() {
        gl_Position = projection * view * model * vec4(aPos, 1.0f);
        FragPos = vec3(view * vec4(aPos, 1.0f));
        vertexColor = vec3(0.94f, 0.9f, 0.69f);
        WorldPos = aPos;
        Normal = normalize(aNormal);
    }
)";

const char* Shaders::fragmentShader = R"(
    #version 460 core
    in vec3 vertexColor;
    in vec3 Normal;
    in vec3 FragPos;
    in vec3 WorldPos;

    out vec4 FragColor;

    uniform bool planeExists;
    uniform vec4 planeData;

    void main() {
        if (planeExists) {
            if (dot(vec3(planeData), WorldPos) + planeData.w >= 0) {
                discard;
            }
        }
        vec3 lightDir = normalize(vec3(0, 0, 0) - FragPos);

        float diff = max(dot(Normal, lightDir), 0.0);
        vec3 diffuse = diff * vec3(0.9, 0.9, 0.9);
        // Ambient + diffuse
        vec3 result = (0.3f + diffuse) * vertexColor; 
        FragColor = vec4(result, 1.0f);
    }
)";

const char* Shaders::viewMatMulCompute = R"(
    #version 460 core
    layout (location = 3) in vec3 aPos;
    layout (std430, binding = 0) buffer gaussianDepth {
        float z_depth[];
    };

    layout (std430, binding = 1) buffer gaussianIndices {
        int gaussian_indices[];
    };

    uniform mat4 view;
    uniform mat4 projection;

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
    out vec3 conic;
    out vec2 coordxy;

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

        mat3 cov3D = computeCov3D(normalize(rotation), scale);

        vec4 view_space = view * vec4(mean, 1);
        vec4 pos2d = projection * view_space;
        // Apply perspective divison here because the 2d coordinates will be 
        // used in other calculations
        pos2d.xyz = pos2d.xyz / pos2d.w;
        pos2d.w = 1.0f;

        vec2 width_height = 2 * hfov_focal.xy * hfov_focal.z;

        // Set limits to avoid extreme perspective distortion & contrain effects of outliers
        float limx = 1.3 * hfov_focal.x;
        float limy = 1.3 * hfov_focal.y;

        float txtz = view_space.x / view_space.z;
        float tytz = view_space.y / view_space.z;

        // Clamped versions of txtz and tytz 
        float tx = min(limx, max(-limx, txtz)) * view_space.z;
        float ty = min(limy, max(-limy, tytz)) * view_space.z; 

        // Cull
        if (any(greaterThan(abs(pos2d.xyz), vec3(1.3)))) {
            gl_Position = vec4(-100, -100, -100, 1);
            return;	
        }

        mat3 J = mat3(
          hfov_focal.z / view_space.z, 0., -(hfov_focal.z * tx) / (view_space.z * view_space.z),
          0., hfov_focal.z / view_space.z, -(hfov_focal.z * ty) / (view_space.z * view_space.z),
          0., 0., 0.
        );
			
        mat3 T = transpose(mat3(view)) * J;
        mat3 cov2d = transpose(T) * transpose(cov3D) * T;

        // A low pass filter according to the paper
        cov2d[0][0] += 0.3f;
        cov2d[1][1] += 0.3f; 

        float det = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];
        if (det == 0.0f)
            gl_Position = vec4(0.f, 0.f, 0.f, 0.f);

        float det_inv = 1.f / det;
        conic = vec3(cov2d[1][1] * det_inv, -cov2d[0][1] * det_inv, cov2d[0][0] * det_inv); 
    
        // Project quad into screen space by doing 3 * standard deviations
        vec2 quadwh_scr = vec2(3.f * sqrt(cov2d[0][0]), 3.f * sqrt(cov2d[1][1]));
        // Convert screenspace quad to NDC
        vec2 quadwh_ndc = quadwh_scr / width_height * 2;

        // Update gaussian's position w.r.t the quad in NDC
        pos2d.xy = pos2d.xy + quadPosition * quadwh_ndc;
        // Calculate where this quad lies in pixel coordinates 
        coordxy = quadPosition * quadwh_scr;

        gl_Position = pos2d;
        outColor = vec3(base_SH * 0.282 + 0.5f);
        opacity = gaussianData[start_index + OPACITY_OFFSET];
    }
)";

const char* Shaders::gaussianFragmentShader = R"(
    #version 460 core

    in vec3 outColor;
    in float opacity;
    in vec3 conic;
    in vec2 coordxy;

    out vec4 FragColor;

    void main() {
        float power = -0.5f * (conic.x * coordxy.x * coordxy.x + conic.z * coordxy.y * coordxy.y) - conic.y * coordxy.x * coordxy.y;
        if(power > 0.0f) discard;
        float alpha = min(0.99f, opacity * exp(power));
        if(alpha < 1.f / 255.f) discard;
        FragColor = vec4(outColor, alpha);
    }
)";
