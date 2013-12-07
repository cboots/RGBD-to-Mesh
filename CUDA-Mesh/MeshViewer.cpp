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


//End platform specific code


MeshViewer* MeshViewer::msSelf = NULL;


MeshViewer::MeshViewer(RGBDDevice* device, int screenwidth, int screenheight)
{
	msSelf = this;
	mDevice = device;
	mWidth = screenwidth;
	mHeight = screenheight;
	mViewState = DISPLAY_MODE_OVERLAY;
}


MeshViewer::~MeshViewer(void)
{
	msSelf = NULL;
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
	initCuda(mXRes, mYRes);

	return initOpenGL(argc, argv);

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

	return DEVICESTATUS_OK;
}



void MeshViewer::initOpenGLHooks()
{
	glutKeyboardFunc(glutKeyboard);
	glutDisplayFunc(glutDisplay);
	glutIdleFunc(glutIdle);
	glutReshapeFunc(glutReshape);	
}


void MeshViewer::initShader()
{
	//Passthrough shaders that sample textures
	const char * color_vert = "shaders/colorVS.glsl";
	const char * color_frag = "shaders/colorFS.glsl";
	const char * depth_vert = "shaders/depthVS.glsl";
	const char * depth_frag = "shaders/depthFS.glsl";

	//Color image shader
	Utility::shaders_t shaders = Utility::loadShaders(color_vert, color_frag);

	color_prog = glCreateProgram();

	glBindAttribLocation(color_prog, quad_attributes::POSITION, "vs_position");
	glBindAttribLocation(color_prog, quad_attributes::TEXCOORD, "vs_texCoord");

	Utility::attachAndLinkProgram(color_prog,shaders);
	

	//DEPTH image shader
	shaders = Utility::loadShaders(depth_vert, depth_frag);

	depth_prog = glCreateProgram();

	glBindAttribLocation(depth_prog, quad_attributes::POSITION, "vs_position");
	glBindAttribLocation(depth_prog, quad_attributes::TEXCOORD, "vs_texCoord");

	Utility::attachAndLinkProgram(depth_prog,shaders);
}


void MeshViewer::initTextures()
{
	glGenTextures(1, &depthTexture);
	glGenTextures(1, &colorTexture);
	glGenTextures(1, &pointCloudTexture);

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

	//Setup point cloud texture
	glBindTexture(GL_TEXTURE_2D, pointCloudTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F , mWidth, mHeight, 0, GL_RGBA, GL_FLOAT,0);

}


void MeshViewer::cleanupTextures()
{
	glDeleteTextures(1, &colorTexture);
	glDeleteTextures(1, &depthTexture);
	glDeleteTextures(1, &pointCloudTexture);
}


void MeshViewer::initPBO()
{
	// set up vertex data parameter

	// Generate a buffer ID called a PBO (Pixel Buffer Object)

	int num_texels = mXRes*mYRes;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLfloat) * num_values;
	glGenBuffers(1,&imagePBO);

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imagePBO);

	// Allocate data for the buffer. 4-channel float image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject( imagePBO);

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
	glVertexAttribPointer(quad_attributes::POSITION, 3, GL_FLOAT, GL_FALSE,sizeof(vertex2_t),0);
	glVertexAttribPointer(quad_attributes::TEXCOORD, 2, GL_FLOAT, GL_FALSE,sizeof(vertex2_t),(void*)sizeof(vec3));
	glEnableVertexAttribArray(quad_attributes::POSITION);
	glEnableVertexAttribArray(quad_attributes::TEXCOORD);

	//indices
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, device_quad.vbo_indices);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6*sizeof(GLushort), indices, GL_STATIC_DRAW);
	device_quad.num_indices = 6;
	//Unplug Vertex Array
	glBindVertexArray(0);
}


//Normalized device coordinates (-1 : 1, -1 : 1) center of viewport, and scale being 
void MeshViewer::drawQuad(GLuint prog, float xNDC, float yNDC, float widthScale, float heightScale, GLuint texture)
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

	//Bind texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glUniform1i(glGetUniformLocation(prog, "u_ColorTex"),0);

	//Draw quad
	glBindVertexArray(device_quad.vertex_array);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, device_quad.vbo_indices);

	glDrawElements(GL_TRIANGLES, device_quad.num_indices, GL_UNSIGNED_SHORT,0);

	glBindVertexArray(0);
}

//Does not return;
void MeshViewer::run()
{
	glutMainLoop();
}


bool MeshViewer::drawColorImageBufferToTexture(GLuint texture)
{
	float4* dptr;
	cudaGLMapBufferObject((void**)&dptr, imagePBO);
	bool result = drawColorImageBufferToPBO(dptr, mXRes, mYRes);
	cudaGLUnmapBufferObject(imagePBO);
	if(result){
		//Draw to texture
		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, imagePBO);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mXRes, mYRes, 
			GL_RGBA, GL_FLOAT, NULL);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	return result;
}

bool MeshViewer::drawDepthImageBufferToTexture(GLuint texture)
{	
	float4* dptr;
	cudaGLMapBufferObject((void**)&dptr, imagePBO);
	bool result = drawDepthImageBufferToPBO(dptr, mXRes, mYRes);
	cudaGLUnmapBufferObject(imagePBO);
	if(result){
		//Draw to texture
		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, imagePBO);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mXRes, mYRes, 
			GL_RGBA, GL_FLOAT, NULL);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	return result;
}

////All the important runtime stuff happens here:
void MeshViewer::display()
{
	ColorPixelArray localColorArray = mColorArray;
	DPixelArray localDepthArray = mDepthArray;

	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//=====CUDA CALLS=====
	//Push buffers
	pushColorArrayToBuffer(localColorArray.get(), mXRes, mYRes);
	pushDepthArrayToBuffer(localDepthArray.get(), mXRes, mYRes);

	//Generate point cloud
	convertToPointCloud();

	//Compute normals
	computePointCloudNormals();

	cudaDeviceSynchronize();
	//=====RENDERING======
	switch(mViewState)
	{
	case DISPLAY_MODE_DEPTH:
		drawDepthImageBufferToTexture(depthTexture);
		glDisable(GL_BLEND);
		drawQuad(depth_prog, 0, 0, 1, 1, depthTexture);
		break;
	case DISPLAY_MODE_IMAGE:
		drawColorImageBufferToTexture(colorTexture);
		glDisable(GL_BLEND);
		drawQuad(color_prog, 0, 0, 1, 1, colorTexture);
		break;
	case DISPLAY_MODE_OVERLAY:
		drawDepthImageBufferToTexture(depthTexture);
		drawColorImageBufferToTexture(colorTexture);


		glDisable(GL_BLEND);
		drawQuad(color_prog, 0, 0, 1, 1, colorTexture);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//Alpha blending
		drawQuad(depth_prog, 0, 0, 1, 1, depthTexture);
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
		}
	}
}

void MeshViewer::onKey(unsigned char key, int /*x*/, int /*y*/)
{
	float newPlayback = 1.0;
	switch (key)
	{
	case 27://ESC
		mDevice->destroyColorStream();
		mDevice->destroyDepthStream();

		mDevice->disconnect();
		mDevice->shutdown();

		cleanupCuda();
		cleanupTextures();
		exit (1);
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
	case('r'):
		cout << "Reloading Shaders" <<endl;
		initShader();
		break;
	}

}


void MeshViewer::reshape(int w, int h)
{
	mWidth = w;
	mHeight = h;
	glBindFramebuffer(GL_FRAMEBUFFER,0);
	glViewport(0,0,(GLsizei)w,(GLsizei)h);
	if (depthTexture != 0 || colorTexture != 0 || pointCloudTexture != 0) {
		cleanupTextures();
	}
	initTextures();
}