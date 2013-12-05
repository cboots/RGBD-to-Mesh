#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;

layout (points) in;
layout (points, max_vertices = 1) out;

in gs_eyeNormal;
in gs_color;

out vec3 fs_eyeNormal;
out vec3 fs_color;


void main()
{

	//Passthrough point	
	fs_eyeNormal = gs_eyeNormal;
	fs_color = gs_color;
	EmitVertex();
    EndPrimitive();
}