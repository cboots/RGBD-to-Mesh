#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform mat4 u_viewInvTrans;
in vec3 vs_position;
in vec3 vs_color;
in vec3 vs_normal;


out vec3 fs_eyeNormal;
out vec3 fs_color;

//Transform each vertex to projection space
void main(void)
{
	gl_Position = u_projMatrix*u_viewMatrix*vec4(vs_position,1.0);
	fs_eyeNormal = normalize(vec3(u_viewInvTrans*vec4(vs_normal,0.0)));
	
	fs_color = vs_color;
}
