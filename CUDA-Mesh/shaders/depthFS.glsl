#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform sampler2D u_ColorTex;

in vec2 fs_texCoord;

out vec4 FragColor;

void main()
{
	//get depth in mm
	float depth = texture(u_ColorTex, fs_texCoord).r;
	
	float shade = 1.0-clamp(depth/5000.0, 0.0, 1.0);
	
	FragColor = step(0.0001, depth)*vec4(shade, 0.0, 0.0, 0.8);
}