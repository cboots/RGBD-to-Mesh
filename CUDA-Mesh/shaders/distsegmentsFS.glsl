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
	vec4 segments = texture(u_Texture0, fs_texCoord*u_TextureScale);
	
	float dist = segments.y/0.015;
	FragColor = vec4(dist+0.5);
	
	if(segments.x < 0)
		FragColor = vec4(0.0);
}