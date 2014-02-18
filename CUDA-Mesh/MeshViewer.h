#pragma once

//Generic includes
#include "Utils.h"

//CUDA GL Includes
#include "cuda_runtime.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include "glslUtility.h"
#include "glm/gtc/matrix_transform.hpp"

//IO Includes
#include <iostream>
#include "FileUtils.h"
#include "FrameLogger.h"
#include "LogDevice.h"

//RGBD Framework includes, device code, and mesh tracking modules
#include "device_structs.h"
#include "RGBDDevice.h"
#include "RGBDFrame.h"
#include "RGBDFrameFactory.h"
#include "MeshTracker.h"
#include "debug_rendering.h"


using namespace glm;

struct Camera{
	vec3 eye;
	vec3 view;
	vec3 up;
	float fovy;
	float zNear;
	float zFar;
};

enum DisplayModes
{
	DISPLAY_MODE_OVERLAY,
	DISPLAY_MODE_DEPTH,
	DISPLAY_MODE_IMAGE,
	DISPLAY_MODE_POINT_CLOUD,
	DISPLAY_MODE_3WAY_DEPTH_IMAGE_OVERLAY
};


class MeshViewer : public RGBDDevice::NewRGBDFrameListener
{
public:

#pragma region Constructors/Destructors
	//CONSTRUCTOR
	MeshViewer(RGBDDevice* device, int screenwidth, int screenheight);

	//DTOR
	~MeshViewer(void);
#pragma endregion

#pragma region Public Functions
	//Initialize CUDA and OpenGL 
	DeviceStatus init(int argc, char **argv);

	//Does not return. Runs main opengl loop
	void run();

	//Event handler
	void onNewRGBDFrame(RGBDFramePtr frame) override;
#pragma endregion

private:

#pragma region Private Variables

	//========General Settings and configuration========

#pragma region General Settings
	//Static self reference. Used for correct opengl callback wrapping
	static MeshViewer* msSelf;

	//Connected modules
	RGBDDevice* mDevice;
	MeshTracker* mMeshTracker;

	//Camera resolution
	int mXRes, mYRes;
	//Screen resolution
	int mWidth, mHeight;
#pragma endregion

	//======STATE VARIABLES=======
#pragma region State Variables
	RGBDFramePtr mLatestFrame;
	ColorPixelArray mColorArray;
	DPixelArray mDepthArray;
	timestamp mLatestTime;

	//FPS Tracking
	time_t seconds;
	int fpstracker;
	float fps;
#pragma endregion

#pragma region Pipeline Options
	FilterMode mFilterMode;

#pragma endregion

	//======Rendering options=======
#pragma region Rendering Options
	//Virtual Camera controls and parameters
	Camera mCamera;
	DisplayModes mViewState;
	bool hairyPoints;
#pragma endregion

	//===========Open GL stuff==============
#pragma region Quad Attributes
	static const GLuint MeshViewer::quadPositionLocation;
	static const GLuint MeshViewer::quadTexcoordsLocation;
	static const char * MeshViewer::quadAttributeLocations[];
	device_mesh2_t device_quad;
#pragma endregion

#pragma region VBO Attributes
	static const GLuint MeshViewer::vbopositionLocation;
	static const GLuint MeshViewer::vbocolorLocation;
	static const GLuint MeshViewer::vbonormalLocation;
	static const char * MeshViewer::vboAttributeLocations[];
#pragma endregion

#pragma region Screen Space VBO Attributes
	static const GLuint PCVBOPositionLocation;//vec3
	static const GLuint PCVBOColorLocation;//vec3
	static const GLuint PCVBONormalLocation;//vec3

	static const GLuint PCVBOStride;//3*vec3
	static const GLuint PCVBO_PositionOffset;
	static const GLuint PCVBO_ColorOffset;
	static const GLuint PCVBO_NormalOffset;
#pragma endregion

	//=========Rendering Variables==========
#pragma region Shader Programs
	//Shader programs
	GLuint depth_prog;
	GLuint color_prog;
	GLuint abs_prog;
	GLuint pcbdebug_prog;
	GLuint pcvbo_prog;
	GLuint pcvbohairy_prog;
	GLuint triangle_prog;
#pragma endregion

#pragma region Buffer Object Indecies
	//PBOs
	GLuint imagePBO0;
	GLuint imagePBO1;
	GLuint imagePBO2;

	GLuint fullscreenPBO;

	//PC VBO
	GLuint pointCloudVBO; 
	GLuint triangleIBO;

	//FBO
	GLuint fullscreenFBO;
#pragma endregion

#pragma region Textures
	//Textures
	//Image space textures
	GLuint colorTexture;
	GLuint depthTexture;
	GLuint positionTexture;
	GLuint normalTexture;

	//Screen space textures
	GLuint FBOColorTexture;
	GLuint FBODepthTexture;
#pragma endregion

#pragma endregion

	//============================PRIVATE FUNCTIONS====================================
#pragma region Private Functions

	//OPENGL Callbacks
#pragma region OPENGL Callbacks

	virtual void display();
	virtual void displayPostDraw(){};	// Overload to draw over the screen image
	virtual void reshape(int w, int h);

	virtual void onKey(unsigned char key, int x, int y);

	static void glutIdle();
	static void glutDisplay();
	static void glutKeyboard(unsigned char key, int x, int y);
	static void glutReshape(int w, int h);
	static void glutMouse(int button, int state, int x, int y);
	static void glutMotion(int x, int y);

	//MOUSE STUFF
	bool dragging;
	bool rightclick;
	int drag_x_last;
	int drag_y_last;
	void mouse_click(int button, int state, int x, int y); 
	void mouse_move(int x, int y);
#pragma endregion

#pragma region OPENGL/CUDA Setup/Teardown functions
	//Open GL Init/cleanup routines
	void initShader();
	void initQuad();
	void initPBO();
	void initFullScreenPBO();
	void initPointCloudVBO();
	void initFBO();
	void cleanupFBO();

	virtual void initRenderingCuda();
	virtual void cleanupRenderingCuda();
	virtual DeviceStatus initOpenGL(int argc, char **argv);
	virtual void initTextures();
	virtual void cleanupTextures();

	void initOpenGLHooks();
#pragma endregion

#pragma region View Settings
	void resetCamera();
#pragma endregion

#pragma region Rendering Functions
	void drawQuad(GLuint prog, float xNDC, float yNDC, float widthScale, float heightScale, GLuint* textures, int numTextures);


	//Draws depth image buffer to the texture.
	//Texture width and height must match the resolution of the depth image.
	//Returns false if width or height does not match, true otherwise
	bool drawDepthImageBufferToTexture(GLuint texture);

	//Draws color image buffer to the texture.
	//Texture width and height must match the resolution of the color image.
	//Returns false if width or height does not match, true otherwise
	bool drawColorImageBufferToTexture(GLuint texture);

	void drawPCBtoTextures(GLuint posTexture, GLuint colTexture, GLuint normTexture);
#pragma endregion


#pragma endregion

};

