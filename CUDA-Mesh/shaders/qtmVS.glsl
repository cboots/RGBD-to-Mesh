#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform mat4 u_viewInvTrans;
uniform mat4 u_modelTransform;
uniform sampler2D u_Texture0;

in vec4 vs_position;

out vec2 fs_texCoord;

//Transform each vertex to projection space
void main(void)
{
	vec4 position = vec4(vs_position.x, vs_position.y, 0.0, 1.0);
	fs_texCoord = vs_position.zw;
	gl_Position = u_projMatrix*u_viewMatrix*u_modelTransform*position;
}
