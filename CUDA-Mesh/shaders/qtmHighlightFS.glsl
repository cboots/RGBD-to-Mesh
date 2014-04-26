#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform mat4 u_viewInvTrans;
uniform mat4 u_modelTransform;
uniform sampler2D u_Texture0;

in vec2 fs_texCoord;

out vec4 FragColor;

void main()
{

	FragColor = vec4(0.0,1.0,0.0,1.0);
}