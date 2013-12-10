#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform mat4 u_viewInvTrans;
in vec3 vs_position;
in vec3 vs_color;
in vec3 vs_normal;


out vec3 gs_eyeNormal;
out vec3 gs_color;

//Transform each vertex to projection space
void main(void)
{
	gl_Position = u_projMatrix*u_viewMatrix*vs_position;
	gs_eyeNormal = vec3(u_viewInvTrans*vec4(vs_up,0.0));
	gs_color = vs_color;
}
