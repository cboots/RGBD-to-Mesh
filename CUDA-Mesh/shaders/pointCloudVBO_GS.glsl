#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;

layout (points) in;
layout (points, max_vertices = 1) out;


in VertexData{
	vec3 EyeNormal;
	vec3 Color;
}vertexData[];

out vec3 fs_eyeNormal;
out vec3 fs_color;


void main()
{

	//Passthrough point	
	fs_eyeNormal = vertexData[0].EyeNormal;
	fs_color = vertexData[0].Color;
	gl_Position = gl_in[0].gl_Position;
	EmitVertex();
    EndPrimitive();
}