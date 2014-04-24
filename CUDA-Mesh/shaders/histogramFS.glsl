#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform sampler2D u_Texture0;
uniform float u_TextureScale;

in vec2 fs_texCoord;

out vec4 FragColor;

void main()
{
	float centerval = texture(u_Texture0, fs_texCoord*u_TextureScale).x;
	FragColor = vec4(centerval/100.0);
	
	if(centerval < 0)
	{
		int peakInd = int(-centerval);
		FragColor = vec4(peakInd & 1, peakInd & 2, peakInd & 4, 1.0);
	}	
}