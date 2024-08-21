#version 330 core

in float frag_x;
in float frag_y;

out vec4 frag_color;

void main()
{
    vec3 color = vec3(frag_x, frag_y, 1.0 - frag_x - frag_y);
    
    frag_color = vec4(color, 1);
}