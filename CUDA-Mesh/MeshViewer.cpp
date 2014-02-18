#include "MeshViewer.h"

//Platform specific code goes here
#include <GL/glew.h>
#include <GL/glut.h>

void MeshViewer::glutIdle()
{
	glutPostRedisplay();
}

void MeshViewer::glutDisplay()
{
	MeshViewer::msSelf->display();
}

void MeshViewer::glutKeyboard(unsigned char key, int x, int y)
{
	MeshViewer::msSelf->onKey(key, x, y);
}


void MeshViewer::glutReshape(int w, int h)
{
	MeshViewer::msSelf->reshape(w, h);
}


void MeshViewer::glutMouse(int button, int state, int x, int y)
{
	MeshViewer::msSelf->mouse_click(button, state, x, y);
}



void MeshViewer::glutMotion(int x, int y)
{
	MeshViewer::msSelf->mouse_move(x, y);
}

//End platform specific code

const GLuint MeshViewer::quadPositionLocation = 0;
const GLuint MeshViewer::quadTexcoordsLocation = 1;
const char * MeshViewer::quadAttributeLocations[] = { "Position", "Texcoords" };

const GLuint MeshViewer::vbopositionLocation = 0;
const GLuint MeshViewer::vbocolorLocation = 1;
const GLuint MeshViewer::vbonormalLocation = 2;
const char * MeshViewer::vboAttributeLocations[] = { "Position", "Color", "Normal" };

const GLuint MeshViewer::PCVBOPositionLocation = 0;//vec3
const GLuint MeshViewer::PCVBOColorLocation = 1;//vec3
const GLuint MeshViewer::PCVBONormalLocation = 2;//vec3

const GLuint MeshViewer::PCVBOStride = 9;//3*vec3
const GLuint MeshViewer::PCVBO_PositionOffset = 0;
const GLuint MeshViewer::PCVBO_ColorOffset = 3;
const GLuint MeshViewer::PCVBO_NormalOffset = 6;

MeshViewer* MeshViewer::msSelf = NULL;


MeshViewer::MeshViewer(RGBDDevice* device, int screenwidth, int screenheight)
{
	msSelf = this;
	mDevice = device;
	mWidth = screenwidth;
	mHeight = screenheight;
	mViewState = DISPLAY_MODE_OVERLAY;
	hairyPoints = false;
	mMeshWireframeMode = false;
	mFastNormals = true;
	mComputeNormals = false;
	mMaxTriangleEdgeLength = 0.1;

	seconds = time (NULL);
	fpstracker = 0;
	fps = 0.0;
	mLatestTime = 0;

	resetCamera();
}


MeshViewer::~MeshViewer(void)
{
	msSelf = NULL;
	if(mMeshTracker != NULL)
		delete mMeshTracker;
}

//Does not return;
void MeshViewer::run()
{
	glutMainLoop();
}

DeviceStatus MeshViewer::init(int argc, char **argv)
{
	//Stream Validation
	if (mDevice->isDepthStreamValid() && mDevice->isColorStreamValid())
	{

		int depthWidth = mDevice->getDepthResolutionX();
		int depthHeight = mDevice->getDepthResolutionY();
		int colorWidth = mDevice->getColorResolutionX();
		int colorHeight = mDevice->getColorResolutionY();

		if (depthWidth == colorWidth &&
			depthHeight == colorHeight)
		{
			mXRes = depthWidth;
			mYRes = depthHeight;

			printf("Color and depth same resolution: D: %dx%d, C: %dx%d\n",
				depthWidth, depthHeight,
				colorWidth, colorHeight);
		}
		else
		{
			printf("Error - expect color and depth to be in same resolution: D: %dx%d, C: %dx%d\n",
				depthWidth, depthHeight,
				colorWidth, colorHeight);
			return DEVICESTATUS_ERROR;
		}
	}
	else if (mDevice->isDepthStreamValid())
	{
		mXRes = mDevice->getDepthResolutionX();
		mYRes = mDevice->getDepthResolutionY();
	}
	else if (mDevice->isColorStreamValid())
	{
		mXRes = mDevice->getColorResolutionX();
		mYRes = mDevice->getColorResolutionY();
	}
	else
	{
		printf("Error - expects at least one of the streams to be valid...\n");
		return DEVICESTATUS_ERROR;
	}

	//Register frame listener
	mDevice->addNewRGBDFrameListener(this);

	//Create mesh tracker
	mMeshTracker = new MeshTracker(mXRes, mYRes);

	//Init rendering cuda code
	initRenderingCuda();

	return initOpenGL(argc, argv);

}

void MeshViewer::initRenderingCuda()
{

}

void MeshViewer::cleanupRenderingCuda()
{

}

DeviceStatus MeshViewer::initOpenGL(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(mWidth, mHeight);
	glutCreateWindow ("CUDA Point Cloud to Mesh");

	//Setup callbacks
	initOpenGLHooks();


	// Init GLEW
	glewInit();
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		// Problem: glewInit failed, something is seriously wrong.
		std::cout << "glewInit failed, aborting." << std::endl;
		return DEVICESTATUS_ERROR;
	}

	//Init elements
	initTextures();
	initShader();
	initQuad();
	initPBO();
	initFullScreenPBO();
	initPointCloudVBO();
	initFBO();

	return DEVICESTATUS_OK;
}



void MeshViewer::initOpenGLHooks()
{
	glutKeyboardFunc(glutKeyboard);
	glutDisplayFunc(glutDisplay);
	glutIdleFunc(glutIdle);
	glutReshapeFunc(glutReshape);	
	glutMouseFunc(glutMouse);
	glutMotionFunc(glutMotion);
}


void MeshViewer::initShader()
{

	const char * pass_vert  = "shaders/passVS.glsl";
	const char * color_frag = "shaders/colorFS.glsl";
	const char * abs_frag = "shaders/absFS.glsl";
	const char * depth_frag = "shaders/depthFS.glsl";
	const char * pcbdebug_frag = "shaders/pointCloudBufferDebugFS.glsl";
	const char * pcvbo_vert = "shaders/pointCloudVBO_VS.glsl";
	const char * pcvbo_geom = "shaders/pointCloudVBO_GS.glsl";
	const char * pcvbo_geom_hairy = "shaders/pointCloudVBOHairy_GS.glsl";
	const char * pcvbo_frag = "shaders/pointCloudVBO_FS.glsl";
	const char * triangle_vert = "shaders/triangle_VS.glsl";
	const char * triangle_frag = "shaders/triangle_FS.glsl";

	//Color image shader
	color_prog = glslUtility::createProgram(pass_vert, NULL, color_frag, quadAttributeLocations, 2);

	//Absolute value shader
	abs_prog = glslUtility::createProgram(pass_vert, NULL, abs_frag, quadAttributeLocations, 2);

	//DEPTH image shader
	depth_prog = glslUtility::createProgram(pass_vert, NULL, depth_frag, quadAttributeLocations, 2);

	//Point Cloud Buffer Debug Shader
	pcbdebug_prog = glslUtility::createProgram(pass_vert, NULL, pcbdebug_frag, quadAttributeLocations, 2);

	//Point cloud VBO renderer
	pcvbo_prog = glslUtility::createProgram(pcvbo_vert, pcvbo_geom, pcvbo_frag, vboAttributeLocations, 3);
	pcvbohairy_prog = glslUtility::createProgram(pcvbo_vert, pcvbo_geom_hairy, pcvbo_frag, vboAttributeLocations, 3);

	triangle_prog = glslUtility::createProgram(triangle_vert, NULL, triangle_frag, vboAttributeLocations, 3);
}


void MeshViewer::initTextures()
{
	//Clear textures
	if (depthTexture != 0 || colorTexture != 0 ||  positionTexture != 0 || normalTexture != 0) {
		cleanupTextures();
	}

	glGenTextures(1, &depthTexture);
	glGenTextures(1, &colorTexture);
	glGenTextures(1, &normalTexture);
	glGenTextures(1, &positionTexture);

	//Setup depth texture
	glBindTexture(GL_TEXTURE_2D, depthTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, mXRes, mYRes, 0, GL_RGBA, GL_FLOAT, 0);

	//Setup color texture
	glBindTexture(GL_TEXTURE_2D, colorTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F , mXRes, mYRes, 0, GL_RGBA, GL_FLOAT,0);

	//Setup position texture
	glBindTexture(GL_TEXTURE_2D, positionTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F , mXRes, mYRes, 0, GL_RGBA, GL_FLOAT,0);

	//Setup normals texture
	glBindTexture(GL_TEXTURE_2D, normalTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F , mXRes, mYRes, 0, GL_RGBA, GL_FLOAT,0);



}


void MeshViewer::cleanupTextures()
{
	//Image space textures
	glDeleteTextures(1, &colorTexture);
	glDeleteTextures(1, &depthTexture);
	glDeleteTextures(1, &positionTexture);
	glDeleteTextures(1, &normalTexture);

}


void checkFramebufferStatus(GLenum framebufferStatus) {
	switch (framebufferStatus) {
	case GL_FRAMEBUFFER_COMPLETE_EXT: break;
	case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
		printf("Attachment Point Unconnected\n");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
		printf("Missing Attachment\n");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
		printf("Dimensions do not match\n");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
		printf("Formats\n");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
		printf("Draw Buffer\n");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
		printf("Read Buffer\n");
		break;
	case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
		printf("Unsupported Framebuffer Configuration\n");
		break;
	default:
		printf("Unkown Framebuffer Object Failure\n");
		break;
	}
}



void MeshViewer::initFBO()
{
	GLenum FBOstatus;
	if(fullscreenFBO != 0)
		cleanupFBO();

	glActiveTexture(GL_TEXTURE9);


	glGenTextures(1, &FBOColorTexture);
	glGenTextures(1, &FBODepthTexture);

	//Set up depth FBO
	glBindTexture(GL_TEXTURE_2D, FBODepthTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, mWidth, mHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);


	//Setup point cloud texture
	glBindTexture(GL_TEXTURE_2D, FBOColorTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F , mWidth, mHeight, 0, GL_RGBA, GL_FLOAT,0);

	glGenFramebuffers(1, &fullscreenFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, fullscreenFBO);
	glViewport(0,0,(GLsizei)mWidth, (GLsizei)mHeight);

	//Bind FBO
	glReadBuffer(GL_NONE);
	GLint color_loc = glGetFragDataLocation(pcvbo_prog,"out_Color");
	GLenum draws [1];
	draws[color_loc] = GL_COLOR_ATTACHMENT0;
	glDrawBuffers(1, draws);


	glBindTexture(GL_TEXTURE_2D, FBODepthTexture);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, FBODepthTexture, 0);
	glBindTexture(GL_TEXTURE_2D, FBOColorTexture);    
	glFramebufferTexture(GL_FRAMEBUFFER, draws[color_loc], FBOColorTexture, 0);

	FBOstatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(FBOstatus != GL_FRAMEBUFFER_COMPLETE) {
		printf("GL_FRAMEBUFFER_COMPLETE failed, CANNOT use FBO[0]\n");
		checkFramebufferStatus(FBOstatus);
	}

	// switch back to window-system-provided framebuffer
	glClear(GL_DEPTH_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void MeshViewer::cleanupFBO()
{

	glDeleteTextures(1,&FBODepthTexture);
	glDeleteTextures(1,&FBOColorTexture);
	glDeleteFramebuffers(1,&fullscreenFBO);
}


void MeshViewer::initPBO()
{
	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	if(imagePBO0){
		glDeleteBuffers(1, &imagePBO0);
	}

	if(imagePBO1){
		glDeleteBuffers(1, &imagePBO1);
	}

	if(imagePBO2){
		glDeleteBuffers(1, &imagePBO2);
	}

	int num_texels = mXRes*mYRes;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLfloat) * num_values;
	glGenBuffers(1,&imagePBO0);
	glGenBuffers(1,&imagePBO1);
	glGenBuffers(1,&imagePBO2);

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imagePBO0);

	// Allocate data for the buffer. 4-channel float image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject( imagePBO0);


	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imagePBO1);

	// Allocate data for the buffer. 4-channel float image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject( imagePBO1);


	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imagePBO2);

	// Allocate data for the buffer. 4-channel float image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject( imagePBO2);
}


void MeshViewer::initFullScreenPBO()
{
	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	if(fullscreenPBO){
		glDeleteBuffers(1, &fullscreenPBO);
	}

	int num_texels = mWidth*mHeight;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLfloat) * num_values;
	glGenBuffers(1,&fullscreenPBO);

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, fullscreenPBO);

	// Allocate data for the buffer. 4-channel float image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject( fullscreenPBO);
}

void MeshViewer::initPointCloudVBO()
{
	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	if(pointCloudVBO){
		glDeleteBuffers(1, &pointCloudVBO);
	}
	if(triangleIBO){
		glDeleteBuffers(1, &triangleIBO);
	}

	//Max num elements
	int max_elements = mWidth*mHeight;
	int size_buf_data = sizeof(PointCloud) * max_elements;

	//Fill with data
	GLfloat *bodies    = new GLfloat[size_buf_data];
	GLuint *indices   = new GLuint[max_elements*3*2];//3 indecies per triangle, 2 tris per pixel(overshoot, but makes initialization simpler)
	for(int i = 0; i < max_elements; i++)
	{
		//Position
		bodies[i*PCVBOStride+0] = 0.0;
		bodies[i*PCVBOStride+1] = 0.0;
		bodies[i*PCVBOStride+2] = -10.0;

		//Color
		bodies[i*PCVBOStride+3] = 1.0;
		bodies[i*PCVBOStride+4] = 1.0;
		bodies[i*PCVBOStride+5] = 0.0;

		//Normal
		bodies[i*PCVBOStride+6] = 0.0;
		bodies[i*PCVBOStride+7] = 0.0;
		bodies[i*PCVBOStride+8] = 0.0;

		indices[i*6+0] = 0;
		indices[i*6+1] = 0;
		indices[i*6+2] = 0;
		indices[i*6+3] = 0;
		indices[i*6+4] = 0;
		indices[i*6+5] = 0;
	}

	glGenBuffers(1,&pointCloudVBO);
	glGenBuffers(1,&triangleIBO);

	glBindBuffer(GL_ARRAY_BUFFER, pointCloudVBO);
	glBufferData(GL_ARRAY_BUFFER, size_buf_data, bodies, GL_DYNAMIC_DRAW);//Initialize


	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangleIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2*3*max_elements*sizeof(GLuint), indices, GL_DYNAMIC_DRAW);

	//Unbind buffers
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	cudaGLRegisterBufferObject( pointCloudVBO);
	cudaGLRegisterBufferObject( triangleIBO);

	delete[] indices;
	delete[] bodies;
}



void MeshViewer::initQuad() {
	vertex2_t verts [] = { {vec3(-1,1,0),vec2(0,0)},
	{vec3(-1,-1,0),vec2(0,1)},
	{vec3(1,-1,0),vec2(1,1)},
	{vec3(1,1,0),vec2(1,0)}};

	unsigned short indices[] = { 0,1,2,0,2,3};

	//Allocate vertex array
	//Vertex arrays encapsulate a set of generic vertex attributes and the buffers they are bound too
	//Different vertex array per mesh.
	glGenVertexArrays(1, &(device_quad.vertex_array));
	glBindVertexArray(device_quad.vertex_array);


	//Allocate vbos for data
	glGenBuffers(1,&(device_quad.vbo_data));
	glGenBuffers(1,&(device_quad.vbo_indices));

	//Upload vertex data
	glBindBuffer(GL_ARRAY_BUFFER, device_quad.vbo_data);
	glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
	//Use of strided data, Array of Structures instead of Structures of Arrays
	glVertexAttribPointer(quadPositionLocation, 3, GL_FLOAT, GL_FALSE,sizeof(vertex2_t),0);
	glVertexAttribPointer(quadTexcoordsLocation, 2, GL_FLOAT, GL_FALSE,sizeof(vertex2_t),(void*)sizeof(vec3));
	glEnableVertexAttribArray(quadPositionLocation);
	glEnableVertexAttribArray(quadTexcoordsLocation);

	//indices
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, device_quad.vbo_indices);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6*sizeof(GLushort), indices, GL_STATIC_DRAW);
	device_quad.num_indices = 6;
	//Unplug Vertex Array
	glBindVertexArray(0);
}


//Normalized device coordinates (-1 : 1, -1 : 1) center of viewport, and scale being 
void MeshViewer::drawQuad(GLuint prog, float xNDC, float yNDC, float widthScale, float heightScale, GLuint* textures, int numTextures)
{
	//Setup program and uniforms
	glUseProgram(prog);
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);

	mat4 persp = mat4(1.0f);//Identity
	mat4 viewmat = mat4(widthScale, 0.0f, 0.0f, 0.0f,
		0.0f, heightScale, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		xNDC, yNDC, 0.0f, 1.0f);


	glUniformMatrix4fv(glGetUniformLocation(prog, "u_projMatrix"),1, GL_FALSE, &persp[0][0] );
	glUniformMatrix4fv(glGetUniformLocation(prog, "u_viewMatrix"),1, GL_FALSE, &viewmat[0][0] );

	//Setup textures
	int location = -1;
	switch(numTextures){
	case 5:
		if ((location = glGetUniformLocation(prog, "u_Texture4")) != -1)
		{
			//has texture
			glActiveTexture(GL_TEXTURE4);
			glBindTexture(GL_TEXTURE_2D, textures[4]);
			glUniform1i(location,4);
		}
	case 4:
		if ((location = glGetUniformLocation(prog, "u_Texture3")) != -1)
		{
			//has texture
			glActiveTexture(GL_TEXTURE3);
			glBindTexture(GL_TEXTURE_2D, textures[3]);
			glUniform1i(location,3);
		}
	case 3:
		if ((location = glGetUniformLocation(prog, "u_Texture2")) != -1)
		{
			//has texture
			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, textures[2]);
			glUniform1i(location,2);
		}
	case 2:
		if ((location = glGetUniformLocation(prog, "u_Texture1")) != -1)
		{
			//has texture
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, textures[1]);
			glUniform1i(location,1);
		}
	case 1:
		if ((location = glGetUniformLocation(prog, "u_Texture0")) != -1)
		{
			//has texture
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textures[0]);
			glUniform1i(location,0);
		}
	}


	//Draw quad
	glBindVertexArray(device_quad.vertex_array);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, device_quad.vbo_indices);

	glDrawElements(GL_TRIANGLES, device_quad.num_indices, GL_UNSIGNED_SHORT,0);

	glBindVertexArray(0);
}




bool MeshViewer::drawColorImageBufferToTexture(GLuint texture)
{
	float4* dptr;
	cudaGLMapBufferObject((void**)&dptr, imagePBO0);
	drawColorImageBufferToPBO(dptr, mMeshTracker->getColorImageDevicePtr(), mXRes, mYRes);
	cudaGLUnmapBufferObject(imagePBO0);
	//Draw to texture
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, imagePBO0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mXRes, mYRes, 
		GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);

	return true;
}

bool MeshViewer::drawDepthImageBufferToTexture(GLuint texture)
{	
	float4* dptr;
	cudaGLMapBufferObject((void**)&dptr, imagePBO0);
	drawDepthImageBufferToPBO(dptr,  mMeshTracker->getDepthImageDevicePtr(), mXRes, mYRes);
	cudaGLUnmapBufferObject(imagePBO0);

	//Draw to texture
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, imagePBO0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mXRes, mYRes, 
		GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);


	return true;
}

void MeshViewer::drawPCBtoTextures(GLuint posTexture, GLuint colTexture, GLuint normTexture)
{
	float4* dptrPosition;
	float4* dptrColor;
	float4* dptrNormal;
	cudaGLMapBufferObject((void**)&dptrPosition, imagePBO0);
	cudaGLMapBufferObject((void**)&dptrColor, imagePBO1);
	cudaGLMapBufferObject((void**)&dptrNormal, imagePBO2);

	drawPCBToPBO(dptrPosition, dptrColor, dptrNormal, mMeshTracker->getPCBDevicePtr(), mXRes, mYRes);

	cudaGLUnmapBufferObject(imagePBO0);
	cudaGLUnmapBufferObject(imagePBO1);
	cudaGLUnmapBufferObject(imagePBO2);

	//Unpack to textures
	glActiveTexture(GL_TEXTURE12);
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, imagePBO2);
	glBindTexture(GL_TEXTURE_2D, normalTexture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mXRes, mYRes, 
		GL_RGBA, GL_FLOAT, NULL);


	glActiveTexture(GL_TEXTURE11);
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, imagePBO1);
	glBindTexture(GL_TEXTURE_2D, colorTexture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mXRes, mYRes, 
		GL_RGBA, GL_FLOAT, NULL);

	glActiveTexture(GL_TEXTURE10);
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, imagePBO0);
	glBindTexture(GL_TEXTURE_2D, positionTexture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mXRes, mYRes, 
		GL_RGBA, GL_FLOAT, NULL);


	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);

}

////All the important runtime stuff happens here:
void MeshViewer::display()
{
	//Grab local copy of latest frames
	ColorPixelArray localColorArray = mColorArray;
	DPixelArray localDepthArray = mDepthArray;

	//Update frame counter
	time_t seconds2 = time (NULL);

	fpstracker++;
	if(seconds2-seconds >= 1){
		fps = fpstracker/(seconds2-seconds);
		fpstracker = 0;
		seconds = seconds2;
	}

	stringstream title;
	title << "RGBD to Mesh Visualization | " << (int)fps  << "FPS";
	glutSetWindowTitle(title.str().c_str());

	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//=====CUDA CALLS=====
	//Push buffers
	mMeshTracker->pushRGBDFrameToDevice(localColorArray, localDepthArray, mLatestTime);

	cudaDeviceSynchronize();


	//=====RENDERING======

	GLuint pcbTextures[] = { positionTexture, colorTexture, normalTexture};
	switch(mViewState)
	{
	case DISPLAY_MODE_DEPTH:
		drawDepthImageBufferToTexture(depthTexture);

		drawQuad(depth_prog, 0, 0, 1, 1, &depthTexture, 1);
		break;
	case DISPLAY_MODE_IMAGE:
		drawColorImageBufferToTexture(colorTexture);

		drawQuad(color_prog, 0, 0, 1, 1, &colorTexture, 1);
		break;
	case DISPLAY_MODE_OVERLAY:
		drawDepthImageBufferToTexture(depthTexture);
		drawColorImageBufferToTexture(colorTexture);


		drawQuad(color_prog, 0, 0, 1, 1, &colorTexture, 1);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//Alpha blending
		drawQuad(depth_prog, 0, 0, 1, 1, &depthTexture, 1);
		glDisable(GL_BLEND);
		break;
	case DISPLAY_MODE_3WAY_DEPTH_IMAGE_OVERLAY:
		drawDepthImageBufferToTexture(depthTexture);
		drawColorImageBufferToTexture(colorTexture);

		drawQuad(color_prog, -0.5, -0.5, 0.5, 0.5, &colorTexture, 1);
		drawQuad(depth_prog, -0.5,  0.5, 0.5, 0.5, &depthTexture, 1);

		drawQuad(color_prog, 0.5, 0, 0.5, 1, &colorTexture, 1);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//Alpha blending
		drawQuad(depth_prog, 0.5, 0, 0.5, 1, &depthTexture, 1);
		glDisable(GL_BLEND);
		break;
	}

	glutSwapBuffers();

}



void MeshViewer::onNewRGBDFrame(RGBDFramePtr frame)
{
	mLatestFrame = frame;
	if(mLatestFrame != NULL)
	{
		if(mLatestFrame->hasColor())
		{
			mColorArray = mLatestFrame->getColorArray();
		}

		if(mLatestFrame->hasDepth())
		{
			mDepthArray = mLatestFrame->getDepthArray();
			mLatestTime = mLatestFrame->getDepthTimestamp();
		}
	}
}

void MeshViewer::onKey(unsigned char key, int /*x*/, int /*y*/)
{
	LogDevice* device = NULL;
	float newPlayback = 1.0;
	vec3 right = vec3(0.0f);

	float cameraHighSpeed = 0.1f;
	float cameraLowSpeed = 0.025f;
	float edgeLengthStep = 0.001f;
	switch (key)
	{
	case 27://ESC
		mDevice->destroyColorStream();
		mDevice->destroyDepthStream();

		mDevice->disconnect();
		mDevice->shutdown();

		cleanupRenderingCuda();
		cleanupTextures();
		exit (0);
		break;
	case '1':
		mViewState = DISPLAY_MODE_OVERLAY;
		break;
	case '2':
		mViewState = DISPLAY_MODE_DEPTH;
		break;
	case '3':
		mViewState = DISPLAY_MODE_IMAGE;
		break;
	case '4':
		mViewState = DISPLAY_MODE_3WAY_DEPTH_IMAGE_OVERLAY;
		break;
	case '5':
		mViewState = DISPLAY_MODE_4WAY_PCB;
		break;
	case '6':
		mViewState = DISPLAY_MODE_POINT_CLOUD;
		break;
	case '7':
		mViewState = DISPLAY_MODE_TRIANGLE;
		break;
	case '8':
		mViewState = DISPLAY_MODE_COLOR_MESH_COMPARE;
		break;
	case '9':
		mViewState = DISPLAY_MODE_POINTCLOUD_MESH_COMPARE;
		break;
	case '0':
		mViewState = DISPLAY_MODE_4WAY_COMPARE;
		break;
	case('r'):
		cout << "Reloading Shaders" <<endl;
		initShader();
		break;
	case('p'):
		cout << "Restarting Playback" << endl;
		device = dynamic_cast<LogDevice*>(mDevice);
		if(device != 0) {
			// old was safely casted to LogDevice
			device->restartPlayback();
		}

		break;
	case '=':
		device = dynamic_cast<LogDevice*>(mDevice);
		if(device != 0) {
			// old was safely casted to LogDevice
			newPlayback = device->getPlaybackSpeed()+0.1;
			cout <<"Playback speed: " << newPlayback << endl;
			device->setPlaybackSpeed(newPlayback);		
		}
		break;
	case '-':
		device = dynamic_cast<LogDevice*>(mDevice);
		if(device != 0) {
			// old was safely casted to LogDevice
			newPlayback = device->getPlaybackSpeed()-0.1;
			cout <<"Playback speed: " << newPlayback << endl;
			device->setPlaybackSpeed(newPlayback);		
		}
		break;
	case 'F':
		mCamera.fovy += 0.5;
		cout << "FOVY :" << mCamera.fovy << endl;
		break;
	case 'f':
		mCamera.fovy -= 0.5;
		cout << "FOVY :" << mCamera.fovy << endl;
		break;
	case 'Q':
		mCamera.eye += cameraLowSpeed*mCamera.up;
		cout << "Camera Eye: " << mCamera.eye.x << " " << mCamera.eye.y << " " << mCamera.eye.z << endl;
		break;
	case 'Z':
		mCamera.eye -= cameraLowSpeed*mCamera.up;
		cout << "Camera Eye: " << mCamera.eye.x << " " << mCamera.eye.y << " " << mCamera.eye.z << endl;
		break;
	case 'W':
		mCamera.eye += cameraLowSpeed*mCamera.view;
		cout << "Camera Eye: " << mCamera.eye.x << " " << mCamera.eye.y << " " << mCamera.eye.z << endl;
		break;
	case 'S':
		mCamera.eye -= cameraLowSpeed*mCamera.view;
		cout << "Camera Eye: " << mCamera.eye.x << " " << mCamera.eye.y << " " << mCamera.eye.z << endl;
		break;
	case 'D':
		right = normalize(cross(mCamera.view, -mCamera.up));
		mCamera.eye += cameraLowSpeed*right;
		cout << "Camera Eye: " << mCamera.eye.x << " " << mCamera.eye.y << " " << mCamera.eye.z << endl;
		break;
	case 'A':
		right = normalize(cross(mCamera.view, -mCamera.up));
		mCamera.eye -= cameraLowSpeed*right;
		cout << "Camera Eye: " << mCamera.eye.x << " " << mCamera.eye.y << " " << mCamera.eye.z << endl;
		break;

	case 'q':
		mCamera.eye += cameraHighSpeed*mCamera.up;
		cout << "Camera Eye: " << mCamera.eye.x << " " << mCamera.eye.y << " " << mCamera.eye.z << endl;
		break;
	case 'z':
		mCamera.eye -= cameraHighSpeed*mCamera.up;
		cout << "Camera Eye: " << mCamera.eye.x << " " << mCamera.eye.y << " " << mCamera.eye.z << endl;
		break;
	case 'w':
		mCamera.eye += cameraHighSpeed*mCamera.view;
		cout << "Camera Eye: " << mCamera.eye.x << " " << mCamera.eye.y << " " << mCamera.eye.z << endl;
		break;
	case 's':
		mCamera.eye -= cameraHighSpeed*mCamera.view;
		cout << "Camera Eye: " << mCamera.eye.x << " " << mCamera.eye.y << " " << mCamera.eye.z << endl;
		break;
	case 'd':
		right = normalize(cross(mCamera.view, -mCamera.up));
		mCamera.eye += cameraHighSpeed*right;
		cout << "Camera Eye: " << mCamera.eye.x << " " << mCamera.eye.y << " " << mCamera.eye.z << endl;
		break;
	case 'a':
		right = normalize(cross(mCamera.view, -mCamera.up));
		mCamera.eye -= cameraHighSpeed*right;
		cout << "Camera Eye: " << mCamera.eye.x << " " << mCamera.eye.y << " " << mCamera.eye.z << endl;
		break;
	case 'x':
		resetCamera();
		cout << "Reset Camera" << endl;
		break;
	case 'h':
		hairyPoints = !hairyPoints;
		cout << "Toggle normal hairs" << endl;
		break;
	case 'M':
		mMaxTriangleEdgeLength += 10*edgeLengthStep;
		cout << "Triangle Max Edge Length (mm): " << mMaxTriangleEdgeLength << endl;
		break;
	case 'm':
		mMaxTriangleEdgeLength += edgeLengthStep;
		cout << "Triangle Max Edge Length (mm): " << mMaxTriangleEdgeLength << endl;
		break;

	case 'N':
		if(mMaxTriangleEdgeLength > 10*edgeLengthStep)
		{
			mMaxTriangleEdgeLength -= 10*edgeLengthStep;
		}
		cout << "Triangle Max Edge Length (mm): " << mMaxTriangleEdgeLength << endl;
		break;
	case 'n':
		if(mMaxTriangleEdgeLength > edgeLengthStep)
		{
			mMaxTriangleEdgeLength -= edgeLengthStep;
		}
		cout << "Triangle Max Edge Length (mm): " << mMaxTriangleEdgeLength << endl;
		break;
	case 'v':
		mMeshWireframeMode = !mMeshWireframeMode;
		cout << "Toggle wireframe mode" << endl;
		break;
	case 'b':
		mFastNormals = !mFastNormals;
		cout << "Fast normals " << (mFastNormals?"On":"Off") << endl;
		break;
	case 'B':
		mComputeNormals = !mComputeNormals;
		cout << "Normal Computation " << (mComputeNormals?"On":"Off") << endl;
		break;
	}

}


void MeshViewer::reshape(int w, int h)
{
	mWidth = w;
	mHeight = h;


	glBindFramebuffer(GL_FRAMEBUFFER,0);
	glViewport(0,0,(GLsizei)w,(GLsizei)h);



	initTextures();
	initFullScreenPBO();//Refresh fullscreen PBO for new resolution
	initFBO();
}



void MeshViewer::resetCamera()
{
	mCamera.eye = vec3(0.0f);
	mCamera.view = vec3(0.0f, 0.0f, -1.0f);
	mCamera.up = vec3(0.0f, 1.0f, 0.0f);
	mCamera.fovy = 23.5f;
	mCamera.zFar = 100.0f;
	mCamera.zNear = 0.01;
}



//MOUSE STUFF
void MeshViewer::mouse_click(int button, int state, int x, int y) {
	if(button == GLUT_LEFT_BUTTON) {
		if(state == GLUT_DOWN) {
			dragging = true;
			drag_x_last = x;
			drag_y_last = y;
		}
		else{
			dragging = false;
		}
	}
	if(button == GLUT_RIGHT_BUTTON) {
		if(state == GLUT_DOWN)
		{
			rightclick = true;
		}else{
			rightclick = false;
		}
	}
}

void MeshViewer::mouse_move(int x, int y) {
	if(dragging) {
		float delX = x-drag_x_last;
		float delY = y-drag_y_last;

		float rotSpeed = 0.1f*PI/180.0f;

		vec3 Up = mCamera.up;
		vec3 Right = normalize(cross(mCamera.view, -mCamera.up));

		if(rightclick)
		{
			mCamera.view = vec3(glm::rotate(glm::rotate(mat4(1.0f), rotSpeed*delY, Right), rotSpeed*delX, Up)*vec4(mCamera.view, 0.0f));
		}else{
			//Simple rotation
			mCamera.view = vec3(glm::rotate(glm::rotate(mat4(1.0f), rotSpeed*delY, Right), rotSpeed*delX, Up)*vec4(mCamera.view, 0.0f));
		}
		drag_x_last = x;
		drag_y_last = y;
	}
}