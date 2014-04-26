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

	vec4 rgbd = texture(u_Texture0, fs_texCoord);
	//Just pass through for now
	FragColor = vec4(rgbd.rgb, 1.0);
	//FragColor = vec4(1.0);
}