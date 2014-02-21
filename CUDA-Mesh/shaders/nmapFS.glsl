#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform sampler2D u_Texture0;
uniform float u_TextureScale;

in vec2 fs_texCoord;

out vec4 FragColor;

void main()
{
	//Just pass through for now
	vec4 color = texture(u_Texture0, fs_texCoord*u_TextureScale);
	FragColor = abs(color);
	
	if(isnan(color.x))
	{
		FragColor = vec4(1.0,1.0,0.0,1.0);
	}
}