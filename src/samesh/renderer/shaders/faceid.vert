#version 330 core

// Vertex Attributes
layout(location = 0) in vec3 position;
layout(location = INST_M_LOC) in mat4 inst_m;

// Uniforms
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

// Outputs
flat out int faceid;

void main()
{
    gl_Position = P * V * M * inst_m * vec4(position, 1.0);
    
    // assumes mesh has been constructed so same face vertices are adjacent and duplicated for each face
    faceid = int(gl_VertexID / 3);
}