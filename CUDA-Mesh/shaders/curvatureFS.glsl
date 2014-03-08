#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform sampler2D u_Texture0;
uniform float u_TextureScale;

in vec2 fs_texCoord;

out vec4 FragColor;

void main()
{
	FragColor = vec4(texture(u_Texture0, fs_texCoord*u_TextureScale).x)*10.0;
	//FragColor = vec4(texture(u_Texture0, fs_texCoord*u_TextureScale).y)/6.28;
	//FragColor = vec4(texture(u_Texture0, fs_texCoord*u_TextureScale).z)/1.57;
	
}