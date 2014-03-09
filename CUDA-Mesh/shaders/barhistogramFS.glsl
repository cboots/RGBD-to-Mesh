#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform sampler2D u_Texture0;
uniform float u_TextureScale;

in vec2 fs_texCoord;

out vec4 FragColor;

void main()
{
	vec2 scaledCoord = fs_texCoord*u_TextureScale;

	vec4 histogramvalue = texture(u_Texture0, scaledCoord)/5000;
	
	FragColor.x = step(1.0-fs_texCoord.y, histogramvalue.x);
	FragColor.y = step(1.0-fs_texCoord.y, histogramvalue.y);
	FragColor.z = step(1.0-fs_texCoord.y, histogramvalue.z);
}