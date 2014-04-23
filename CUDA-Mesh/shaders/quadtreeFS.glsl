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
	float quadtreeVal = texture(u_Texture0, fs_texCoord*u_TextureScale).x;
	
	if(isnan(quadtreeVal))
	{
		FragColor = 0.1*vec4(1.0);
	}else{
		int quadTreeDegree = int(quadtreeVal);
		if(quadTreeDegree < 0)
		{
			FragColor = vec4(0.0);
		}else if(quadTreeDegree == 0)
		{
			FragColor = 0.1*vec4(1.0);
		}else{
			FragColor = vec4((quadTreeDegree >> 6) & 0x7,(quadTreeDegree >> 3) & 0x7,quadTreeDegree & 0x7,1.0);
		}
	}
}