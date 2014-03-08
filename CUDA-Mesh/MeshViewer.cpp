#include "MeshViewer.h"

//Platform specific code goes here
#include <GL/glew.h>
#include <GL/glut.h>

#pragma region GLUT Hooks
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

#pragma endregion

//End platform specific code

#pragma region Variable definitions
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

const GLuint MeshViewer::PCVBOStride = 9;//3*float3
const GLuint MeshViewer::PCVBO_PositionOffset = 0;
const GLuint MeshViewer::PCVBO_ColorOffset = 3;
const GLuint MeshViewer::PCVBO_NormalOffset = 6;

MeshViewer* MeshViewer::msSelf = NULL;
#pragma endregion


#pragma region Constructors/Destructors
MeshViewer::MeshViewer(RGBDDevice* device, int screenwidth, int screenheight)
{
	//Setup general modules
	msSelf = this;
	mDevice = device;
	mWidth = screenwidth;
	mHeight = screenheight;

	//Setup default rendering/pipeline settings
	mFilterMode = BILATERAL_FILTER;
	mNormalMode = AVERAGE_GRADIENT_NORMALS;
	mViewState = DISPLAY_MODE_OVERLAY;
	hairyPoints = false;
	mSpatialSigma = 2.0f;
	mDepthSigma = 0.005f;
	mMaxDepth = 6.0f;

	seconds = time (NULL);
	fpstracker = 0;
	fps = 0.0;
	mLatestTime = 0;
	mLastSubmittedTime = 0;

	resetCamera();
}


MeshViewer::~MeshViewer(void)
{
	msSelf = NULL;
	if(mMeshTracker != NULL)
		delete mMeshTracker;
}

#pragma endregion


//Does not return;
void MeshViewer::run()
{
	glutMainLoop();
}

#pragma region Helper functions
//Framebuffer status helper function
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
#pragma endregion

#pragma region Init/Cleanup Functions

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

	//Create mesh tracker and set default values
	mMeshTracker = new MeshTracker(mXRes, mYRes, mDevice->getColorIntrinsics());
	mMeshTracker->setGaussianSpatialSigma(mSpatialSigma);

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
	const char * vmap_frag = "shaders/vmapFS.glsl";
	const char * nmap_frag = "shaders/nmapFS.glsl";
	const char * curvature_frag = "shaders/curvatureFS.glsl";
	const char * histogram_frag = "shaders/histogramFS.glsl";

	//Color image shader
	color_prog = glslUtility::createProgram(pass_vert, NULL, color_frag, quadAttributeLocations, 2);

	//Absolute value shader
	abs_prog = glslUtility::createProgram(pass_vert, NULL, abs_frag, quadAttributeLocations, 2);

	//DEPTH image shader
	depth_prog = glslUtility::createProgram(pass_vert, NULL, depth_frag, quadAttributeLocations, 2);

	//VMap display debug shader
	vmap_prog = glslUtility::createProgram(pass_vert, NULL, vmap_frag, quadAttributeLocations, 2);

	//NMap display debug shader
	nmap_prog = glslUtility::createProgram(pass_vert, NULL, nmap_frag, quadAttributeLocations, 2);

	curvemap_prog = glslUtility::createProgram(pass_vert, NULL, curvature_frag, quadAttributeLocations, 2);

	histogram_prog = glslUtility::createProgram(pass_vert, NULL, histogram_frag, quadAttributeLocations, 2);
}

void MeshViewer::initTextures()
{
	//Clear textures
	if (texture0 != 0 || texture1 != 0 ||  texture2 != 0 || texture3 != 0) {
		cleanupTextures();
	}

	glGenTextures(1, &texture0);
	glGenTextures(1, &texture1);
	glGenTextures(1, &texture2);
	glGenTextures(1, &texture3);

	//Setup Texture 0
	glBindTexture(GL_TEXTURE_2D, texture0);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, mXRes, mYRes, 0, GL_RGBA, GL_FLOAT, 0);

	//Setup Texture 1
	glBindTexture(GL_TEXTURE_2D, texture1);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F , mXRes, mYRes, 0, GL_RGBA, GL_FLOAT,0);

	//Setup Texture 2
	glBindTexture(GL_TEXTURE_2D, texture2);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F , mXRes, mYRes, 0, GL_RGBA, GL_FLOAT,0);

	//Setup Texture 3
	glBindTexture(GL_TEXTURE_2D, texture3);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F , mXRes, mYRes, 0, GL_RGBA, GL_FLOAT,0);



}

void MeshViewer::cleanupTextures()
{
	//Image space textures
	glDeleteTextures(1, &texture0);
	glDeleteTextures(1, &texture1);
	glDeleteTextures(1, &texture2);
	glDeleteTextures(1, &texture3);

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

	//TODO: Bind FBO to programs
	/*
	glReadBuffer(GL_NONE);
	GLint color_loc = glGetFragDataLocation(pcvbo_prog,"out_Color");
	GLenum draws [1];
	draws[color_loc] = GL_COLOR_ATTACHMENT0;
	glDrawBuffers(1, draws);

	glBindTexture(GL_TEXTURE_2D, FBODepthTexture);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, FBODepthTexture, 0);
	glBindTexture(GL_TEXTURE_2D, FBOColorTexture);    
	glFramebufferTexture(GL_FRAMEBUFFER, draws[color_loc], FBOColorTexture, 0);

	*/
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


#pragma endregion

#pragma region Rendering Helper Functions
//Normalized device coordinates (-1 : 1, -1 : 1) center of viewport, and scale being 
void MeshViewer::drawQuad(GLuint prog, float xNDC, float yNDC, float widthScale, float heightScale, float textureScale, GLuint* textures, int numTextures)
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
	if ((location = glGetUniformLocation(prog, "u_TextureScale")) != -1)
	{
		//has texture scale parameter
		glUniform1f(location,textureScale);
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

void MeshViewer::drawVMaptoTexture(GLuint texture, int level)
{
	float4* dptrVMap;
	cudaGLMapBufferObject((void**)&dptrVMap, imagePBO0);

	clearPBO(dptrVMap, mXRes, mYRes, 0.0f);
	drawVMaptoPBO(dptrVMap, mMeshTracker->getVMapPyramid(), level, mXRes, mYRes);

	cudaGLUnmapBufferObject(imagePBO0);

	//Unpack to textures
	glActiveTexture(GL_TEXTURE12);
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, imagePBO0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mXRes, mYRes, 
		GL_RGBA, GL_FLOAT, NULL);

	//Unbind buffers
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);

}

void MeshViewer::drawNMaptoTexture(GLuint texture, int level)
{
	float4* dptrNMap;
	cudaGLMapBufferObject((void**)&dptrNMap, imagePBO0);

	clearPBO(dptrNMap, mXRes, mYRes, 0.0f);
	drawNMaptoPBO(dptrNMap, mMeshTracker->getNMapPyramid(), level, mXRes, mYRes);

	cudaGLUnmapBufferObject(imagePBO0);

	//Unpack to textures
	glActiveTexture(GL_TEXTURE12);
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, imagePBO0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mXRes, mYRes, 
		GL_RGBA, GL_FLOAT, NULL);

	//Unbind buffers
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);

}

void MeshViewer::drawCurvaturetoTexture(GLuint texture)
{
	float4* dptrNMap;
	cudaGLMapBufferObject((void**)&dptrNMap, imagePBO0);

	clearPBO(dptrNMap, mXRes, mYRes, 0.0f);
	drawCurvaturetoPBO(dptrNMap, mMeshTracker->getCurvature(), mXRes, mYRes);

	cudaGLUnmapBufferObject(imagePBO0);

	//Unpack to textures
	glActiveTexture(GL_TEXTURE12);
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, imagePBO0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mXRes, mYRes, 
		GL_RGBA, GL_FLOAT, NULL);

	//Unbind buffers
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);
}


void MeshViewer::drawNormalHistogramtoTexture(GLuint texture)
{
	float4* dptrNMap;
	cudaGLMapBufferObject((void**)&dptrNMap, imagePBO0);

	clearPBO(dptrNMap, mXRes, mYRes, 0.0f);
	drawNormalVoxelsToPBO(dptrNMap, mMeshTracker->getDeviceNormalHistogram(), mXRes, mYRes, 
		mMeshTracker->getNormalXSubdivisions(), mMeshTracker->getNormalYSubdivisions());

	cudaGLUnmapBufferObject(imagePBO0);

	//Unpack to textures
	glActiveTexture(GL_TEXTURE12);
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, imagePBO0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mXRes, mYRes, 
		GL_RGBA, GL_FLOAT, NULL);

	//Unbind buffers
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);

}


void MeshViewer::drawRGBMaptoTexture(GLuint texture, int level)
{
	float4* dptrRGBMap;
	cudaGLMapBufferObject((void**)&dptrRGBMap, imagePBO0);

	clearPBO(dptrRGBMap, mXRes, mYRes, 0.0f);
	drawRGBMaptoPBO(dptrRGBMap, mMeshTracker->getRGBMapSOA(), level, mXRes, mYRes);

	cudaGLUnmapBufferObject(imagePBO0);

	//Unpack to textures
	glActiveTexture(GL_TEXTURE12);
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, imagePBO0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mXRes, mYRes, 
		GL_RGBA, GL_FLOAT, NULL);

	//Unbind buffers
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);
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

#pragma endregion

#pragma region Event Handlers
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

#pragma endregion

void MeshViewer::display()
{
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

	//=====Tracker Pipeline=====
	//Check if log playback has restarted (special edge case)
	if(mLastSubmittedTime > mLatestTime){
		//Maybe stream has restarted playback?
		cout << "Reseting tracking, because timestamp" << endl;
		mMeshTracker->resetTracker();
		mLastSubmittedTime = 0;
	}

	//Check if we have a new frame
	if(mLatestTime > mLastSubmittedTime)
	{
		//Now we have new data, so run pipeline
		mLastSubmittedTime = mLatestTime;

		//Grab local copy of latest frames
		ColorPixelArray localColorArray = mColorArray;
		DPixelArray localDepthArray = mDepthArray;

		//Push buffers
		mMeshTracker->pushRGBDFrameToDevice(localColorArray, localDepthArray, mLatestTime);

		cudaDeviceSynchronize();

		mMeshTracker->buildRGBSOA();

		switch(mFilterMode)
		{
		case BILATERAL_FILTER:
			mMeshTracker->buildVMapBilateralFilter(mMaxDepth, mDepthSigma);
			break;
		case GAUSSIAN_FILTER:
			mMeshTracker->buildVMapGaussianFilter(mMaxDepth);
			break;
		case NO_FILTER:
		default:
			mMeshTracker->buildVMapNoFilter(mMaxDepth);
			break;

		}

		switch(mNormalMode)
		{
		case SIMPLE_NORMALS:
			mMeshTracker->buildNMapSimple();
			mMeshTracker->estimateCurvatureFromNormals();
			break;
		case AVERAGE_GRADIENT_NORMALS:
			mMeshTracker->buildNMapAverageGradient(4);
			mMeshTracker->estimateCurvatureFromNormals();
			break;
		case PCA_NORMALS:
			mMeshTracker->buildNMapPCA(0.03f); 
			break;
		}

		//Launch kernels for subsampling
		mMeshTracker->subsamplePyramids();

		mMeshTracker->GPUSimpleSegmentation();


	}//=====End of pipeline code=====


	//=====RENDERING======
	switch(mViewState)
	{
	case DISPLAY_MODE_DEPTH:
		drawDepthImageBufferToTexture(texture0);

		drawQuad(depth_prog, 0, 0, 1, 1, 1.0, &texture0, 1);
		break;
	case DISPLAY_MODE_IMAGE:
		drawColorImageBufferToTexture(texture1);

		drawQuad(color_prog, 0, 0, 1, 1, 1.0, &texture1, 1);
		break;
	case DISPLAY_MODE_OVERLAY:
		drawDepthImageBufferToTexture(texture0);
		drawColorImageBufferToTexture(texture1);


		drawQuad(color_prog, 0, 0, 1, 1, 1.0, &texture1, 1);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//Alpha blending
		drawQuad(depth_prog, 0, 0, 1, 1, 1.0, &texture0, 1);
		glDisable(GL_BLEND);
		break;
	case DISPLAY_MODE_3WAY_DEPTH_IMAGE_OVERLAY:
		drawDepthImageBufferToTexture(texture0);
		drawColorImageBufferToTexture(texture1);

		drawQuad(color_prog, -0.5, -0.5, 0.5, 0.5, 1.0, &texture1, 1);
		drawQuad(depth_prog, -0.5,  0.5, 0.5, 0.5, 1.0, &texture0, 1);

		drawQuad(color_prog, 0.5, 0, 0.5, 1, 1.0, &texture1, 1);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//Alpha blending
		drawQuad(depth_prog, 0.5, 0, 0.5, 1, 1.0, &texture0, 1);
		glDisable(GL_BLEND);
		break;
	case DISPLAY_MODE_HISTOGRAM_DEBUG:
		drawDepthImageBufferToTexture(texture0);
		drawColorImageBufferToTexture(texture1);
		drawNMaptoTexture(texture2, 0);
		drawNormalHistogramtoTexture(texture3);

		drawQuad(depth_prog,  0.5,  0.5, 0.5, 0.5, 1.0, &texture0, 1);//UR depth
		drawQuad(color_prog,  0.5, -0.5, 0.5, 0.5, 1.0,  &texture1, 1);//LR color
		drawQuad(nmap_prog, -0.5, -0.5, 0.5, 0.5, 1.0,  &texture2, 1);//LL normal 
		drawQuad(histogram_prog, -0.5,  0.5, 0.5, 0.5, 0.6,  &texture3, 1);//UL histogram
		break;
	case DISPLAY_MODE_NMAP_DEBUG:
		drawNMaptoTexture(texture0, 0);
		drawNMaptoTexture(texture1, 1);
		drawNMaptoTexture(texture2, 2);
		drawColorImageBufferToTexture(texture3);

		drawQuad(nmap_prog,  0.5,  0.5, 0.5, 0.5, 1.0, &texture0, 1);//UR Level0 NMap
		drawQuad(nmap_prog,  0.5, -0.5, 0.5, 0.5, 0.5,  &texture1, 1);//LR Level1 NMap
		drawQuad(nmap_prog, -0.5, -0.5, 0.5, 0.5, 0.25,  &texture2, 1);//LL Level2 NMap
		drawQuad(color_prog, -0.5,  0.5, 0.5, 0.5, 1.0,  &texture3, 1);//UL Original depth
		break;

	case DISPLAY_MODE_VMAP_DEBUG:
		drawVMaptoTexture(texture0, 0);
		drawVMaptoTexture(texture1, 1);
		drawVMaptoTexture(texture2, 2);
		drawDepthImageBufferToTexture(texture3);

		drawQuad(vmap_prog,  0.5,  0.5, 0.5, 0.5, 1.0, &texture0, 1);//UR Level0 VMap
		drawQuad(vmap_prog,  0.5, -0.5, 0.5, 0.5, 0.5,  &texture1, 1);//LR Level1 VMap
		drawQuad(vmap_prog, -0.5, -0.5, 0.5, 0.5, 0.25,  &texture2, 1);//LL Level2 VMap
		drawQuad(depth_prog, -0.5,  0.5, 0.5, 0.5, 1.0,  &texture3, 1);//UL Original depth
		break;
	case DISPLAY_MODE_CURVATURE_DEBUG:

		drawDepthImageBufferToTexture(texture0);
		drawColorImageBufferToTexture(texture1);
		drawNMaptoTexture(texture2, 0);
		drawCurvaturetoTexture(texture3);

		drawQuad(depth_prog,  0.5,  0.5, 0.5, 0.5, 1.0, &texture0, 1);//UR depth
		drawQuad(color_prog,  0.5, -0.5, 0.5, 0.5, 1.0,  &texture1, 1);//LR color
		drawQuad(nmap_prog, -0.5, -0.5, 0.5, 0.5, 1.0,  &texture2, 1);//LL normal 
		drawQuad(curvemap_prog, -0.5,  0.5, 0.5, 0.5, 1.0,  &texture3, 1);//UL curvature
		break;
	}

	glutSwapBuffers();

}

#pragma region OpenGL Callbacks
////All the important runtime stuff happens here:

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
		cudaDeviceReset();
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
		mViewState = DISPLAY_MODE_HISTOGRAM_DEBUG;
		break;
	case '6':
		mViewState = DISPLAY_MODE_VMAP_DEBUG;
		break;
	case '7':
		mViewState = DISPLAY_MODE_NMAP_DEBUG;
		break;
	case '8':
		mViewState = DISPLAY_MODE_CURVATURE_DEBUG;
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
	case 'g':
		mFilterMode = GAUSSIAN_FILTER;
		cout << "Gaussian Filter" << endl;
		break;
	case 'b':
		mFilterMode = BILATERAL_FILTER;
		cout << "Bilateral Filter" << endl;
		break;
	case 'n':
		mFilterMode = NO_FILTER;
		cout << "No Filter" << endl;
		break;
	case '[':
		mSpatialSigma -= 0.1f;
		cout << "Spatial Sigma = " << mSpatialSigma << " Lateral Pixels" << endl;
		mMeshTracker->setGaussianSpatialSigma(mSpatialSigma);
		break;
	case ']':
		mSpatialSigma += 0.1f;
		cout << "Spatial Sigma = " << mSpatialSigma << " Lateral Pixels" << endl;
		mMeshTracker->setGaussianSpatialSigma(mSpatialSigma);
		break;
	case '{':
		mDepthSigma -= 0.005f;
		cout << "Depth Sigma = " << mDepthSigma << " (m)" << endl;
		break;
	case '}':
		mDepthSigma += 0.005f;
		cout << "Depth Sigma = " << mDepthSigma << " (m)" << endl;
		break;
	case '\'':
		mMaxDepth += 0.25f;
		cout << "Max Depth: " << mMaxDepth << " (m)" << endl;
		break;
	case ';':
		mMaxDepth -= 0.25f;
		cout << "Max Depth: " << mMaxDepth << " (m)" << endl;
		break;
	case ',':
		mNormalMode = AVERAGE_GRADIENT_NORMALS;
		cout << "Average Gradient Normals Mode"<< endl;
		break;
	case '.':
		mNormalMode = SIMPLE_NORMALS;
		cout << "Simple Normals Mode"<< endl;
		break;
	case '/':
		mNormalMode = PCA_NORMALS;
		cout << "PCA Normals Mode" << endl;
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


#pragma region OpenGL Mouse Callbacks

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

#pragma endregion

#pragma endregion