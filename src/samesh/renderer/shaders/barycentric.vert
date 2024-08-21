#version 330 core

// Vertex Attributes
layout(location = 0) in vec3 position;
layout(location = INST_M_LOC) in mat4 inst_m;

// Uniforms
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

// Outputs
out float frag_x;
out float frag_y;

void main()
{
    gl_Position = P * V * M * inst_m * vec4(position, 1.0);
    
    // assumes mesh has been constructed so same face vertices are adjacent and duplicated for each face
    frag_x = float(((uint(gl_VertexID) + 1u) % 3u) % 2u); 
    frag_y = float(((uint(gl_VertexID) + 0u) % 3u) % 2u); 
}