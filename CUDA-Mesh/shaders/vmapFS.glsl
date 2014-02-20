#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform sampler2D u_Texture0;
uniform float u_TextureScale;

in vec2 fs_texCoord;

out vec4 FragColor;

void main()
{
	//get depth in m
	float depth = texture(u_Texture0, fs_texCoord*u_TextureScale).z;
	
	float shade = 1.0-clamp(depth/(10.0), 0.0, 1.0);
	
	FragColor = step(0.0001, depth)*vec4(shade, 1.0-step(0.0001,shade), 1.0-shade, 0.7);
	
}