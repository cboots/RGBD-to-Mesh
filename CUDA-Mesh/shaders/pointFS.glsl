#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;


in vec3 fs_eyeNormal;
in vec3 fs_color;


out vec4 FragColor;

void main()
{
	//Just pass through for now
	FragColor = vec4(fs_color, 1.0);
}