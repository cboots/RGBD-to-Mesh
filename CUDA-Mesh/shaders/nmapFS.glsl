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
	vec4 norm = texture(u_Texture0, fs_texCoord*u_TextureScale);
	FragColor = abs(norm);
	FragColor.a = 1.0;
	
	vec2 texCoord = fs_texCoord*u_TextureScale;
	float dxy = 0.002;
	vec4 normC = texture(u_Texture0, texCoord);
	float dot1 = dot(normC, texture(u_Texture0, texCoord + vec2(0.0, dxy)));
	float dot2 = dot(normC, texture(u_Texture0, texCoord + vec2(dxy, 0.0)));
	
	float inplane = step(0.95, min(dot1,dot2));
	FragColor = vec4(inplane, 0.0, min(dot1,dot2), 1.0);
	
	
	/*
	if(isnan(norm.x))
	{
		FragColor = vec4(1.0,1.0,0.0,1.0);
	}
	*/
}