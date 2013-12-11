#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;

uniform sampler2D u_Texture0;//Position
uniform sampler2D u_Texture1;//Color
uniform sampler2D u_Texture2;//Normal


in vec2 fs_texCoord;

out vec4 FragColor;

void main()
{
	vec4 position = texture(u_Texture0, fs_texCoord);
	vec4 color = 	texture(u_Texture1, fs_texCoord);
	vec4 normal = 	texture(u_Texture2, fs_texCoord);
	FragColor = vec4(-position.y, position.y, 0.0, 1.0);
	//FragColor = vec4(-position.zzz/10.0, 1.0);
	//FragColor = vec4(position.x, position.y, -position.z, 1.0);
	//FragColor = abs(normal);
	
}