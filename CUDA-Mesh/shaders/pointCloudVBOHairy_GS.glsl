#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;

layout (points) in;
layout (line_strip, max_vertices = 2) out;


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
	
	fs_eyeNormal = vertexData[0].EyeNormal;
	fs_color = vertexData[0].Color;
	gl_Position = gl_in[0].gl_Position+vec4(fs_eyeNormal*0.05,0.0);
	EmitVertex();
	
    EndPrimitive();
}