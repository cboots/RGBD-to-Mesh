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
	
	float depth = -position.z;
	float shade = 1.0-clamp(mod(depth,0.005)/0.005, 0.0, 1.0);
	FragColor = step(0.0001, depth)*vec4(shade, 1.0-step(0.0001,shade), 1.0-shade, 0.7);
	//FragColor = vec4(-position.y, position.y, 0.0, 1.0);
	//FragColor = vec4(-position.zzz/10.0, 1.0);
	//FragColor = vec4(step(0.01, position.z)*(1.0-step(0.01, length(normal))), 0.0, 0.0, 1.0);
	//FragColor = vec4(position.x, position.y, -position.z, 1.0);
	//
	//if (normal.x != 0 || normal.y != 0 || normal.z != 0) {
	//	FragColor = normal;
	//} else {
	//	FragColor = vec4(vec3(1.0f, 0.0f, 0.0f), 1.0f);
	//}
	//
	//FragColor = vec4(1.0f);
	//FragColor = vec4(0.0f, 1.0e9f*max(normal.x, max(normal.y, normal.z)), 0.0f, 1.0f);
	//FragColor = abs(normal);
}