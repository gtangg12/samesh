#version 330 core

flat in int faceid;

out vec4 frag_color;

void main()
{
    int x = faceid / 65536;
    int r = faceid % 65536;
    int y = r / 256;
    int z = r % 256;
    // Create a color based on the normalized ID
    vec3 color = vec3(float(x / 255.0), float(y / 255.0), float(z / 255.0));
    
    frag_color = vec4(color, 1);
}