#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform sampler2D u_Texture0;
uniform float u_TextureScale;

in vec2 fs_texCoord;

out vec4 FragColor;

void main()
{
	/*
	float curvature = texture(u_Texture0, fs_texCoord*u_TextureScale).x;
	FragColor = vec4(1.0-step(0.03,curvature),0.0,0.0,1.0);
	
	if(isnan(curvature))
	{
		FragColor = vec4(1.0,1.0,0.0,1.0);
	}
	*/
	
	FragColor = vec4(texture(u_Texture0, fs_texCoord*u_TextureScale).z)/3.14;
	
	
}