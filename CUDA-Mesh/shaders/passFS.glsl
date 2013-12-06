#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform sampler2D u_ColorTex;

in vec2 fs_texCoord;

out vec4 FragColor;

void main()
{
	//Just pass through for now
	//FragColor = texture(u_ColorTex, fs_texCoord);
	FragColor = vec4(fs_texCoord.x, fs_texCoord.y, 0.0, 1.0);
}