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
	FragColor = vec4(centerval/2500.0);
	
	/*
	float step = 1.0/32.0;
	vec4 neighbors = vec4(	texture(u_Texture0, (fs_texCoord+vec2(step,0.0))*u_TextureScale).x,
							texture(u_Texture0, (fs_texCoord+vec2(0.0,step))*u_TextureScale).x,
							texture(u_Texture0, (fs_texCoord+vec2(-step,0.0))*u_TextureScale).x,
							texture(u_Texture0, (fs_texCoord+vec2(0.0,-step))*u_TextureScale).x);

	if(centerval > 500)
		if(centerval > max(max(max(neighbors.x, neighbors.y), neighbors.z),neighbors.w))
			FragColor = vec4(1.0, 0.0, 0.0, 1.0);
		*/
	if(centerval > 1000000)
		FragColor = vec4(1.0, 0.0, 0.0, 1.0);
			
}