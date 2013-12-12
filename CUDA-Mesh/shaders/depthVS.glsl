#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;

uniform sampler2D u_Texture0;

in vec3 vs_position;
in vec2 vs_texCoord;

out vec2 fs_texCoord;

//Transform each vertex to projection space
void main(void)
{
	gl_Position = u_projMatrix*u_viewMatrix*vec4(vs_position, 1.0);
	fs_texCoord = vs_texCoord;
}
