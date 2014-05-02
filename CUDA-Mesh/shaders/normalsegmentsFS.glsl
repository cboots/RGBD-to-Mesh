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
	
	int seg = int(segments.x);
	switch(seg)
	{
	case 0:
		FragColor = vec4(1,0,0,1);
		break;
	case 8:
		FragColor = vec4(0,1,0,1);
		break;
	case 16:
		FragColor = vec4(1,1,0,1);
		break;
	default:
		FragColor = vec4(0,0,0,1);
	}		
	//FragColor = vec4(seg);
}