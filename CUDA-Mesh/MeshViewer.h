#pragma once


#include <iostream>
#include "RGBDDevice.h"
#include "RGBDFrame.h"
#include "RGBDFrameFactory.h"
#include "FileUtils.h"
#include "FrameLogger.h"
#include "device_structs.h"
#include "Device.h"
#include "glslUtility.h"
#include "LogDevice.h"

using namespace glm;

enum DisplayModes
{
	DISPLAY_MODE_OVERLAY,
	DISPLAY_MODE_DEPTH,
	DISPLAY_MODE_IMAGE,
	DISPLAY_MODE_POINT_CLOUD,
	DISPLAY_MODE_3WAY_DEPTH_IMAGE_OVERLAY,
	DISPLAY_MODE_4WAY_PCB
};




class MeshViewer : public RGBDDevice::NewRGBDFrameListener
{
private:

	static const GLuint MeshViewer::quadPositionLocation;
	static const GLuint MeshViewer::quadTexcoordsLocation;
	static const char * MeshViewer::quadAttributeLocations[];

	static const GLuint MeshViewer::vbopositionLocation;
	static const GLuint MeshViewer::vbocolorLocation;
	static const GLuint MeshViewer::vbonormalLocation;
	static const char * MeshViewer::vboAttributeLocations[];


public:
	MeshViewer(RGBDDevice* device, int screenwidth, int screenheight);
	~MeshViewer(void);

	DeviceStatus init(int argc, char **argv);

	//Does not return
	void run();

	//Event handler
	void onNewRGBDFrame(RGBDFramePtr frame) override;
protected:
	//Display functions
	virtual void display();
	virtual void displayPostDraw(){};	// Overload to draw over the screen image
	virtual void reshape(int w, int h);

	virtual void onKey(unsigned char key, int x, int y);

	virtual DeviceStatus initOpenGL(int argc, char **argv);
	virtual void initTextures();
	virtual void cleanupTextures();

	void initOpenGLHooks();

	//Textures
	//Image space textures
	GLuint colorTexture;
	GLuint depthTexture;
	GLuint positionTexture;
	GLuint normalTexture;

	//Screen space textures
	GLuint pointCloudTexture;

	device_mesh2_t device_quad;
	static MeshViewer* msSelf;
	RGBDDevice* mDevice;
	int mXRes, mYRes;
	int mWidth, mHeight;

	DisplayModes mViewState;

	RGBDFramePtr mLatestFrame;
	ColorPixelArray mColorArray;
	DPixelArray mDepthArray;

	//Shader programs
	GLuint depth_prog;
	GLuint color_prog;
	GLuint pcbdebug_prog;
	GLuint pcvbo_prog;

	//PBOs
	GLuint imagePBO0;
	GLuint imagePBO1;
	GLuint imagePBO2;

	GLuint fullscreenPBO;

	//PC VBO
	GLuint pointCloudVBO; 


	void initShader();
	void initQuad();
	void initPBO();
	void initFullScreenPBO();
	void initPointCloudVBO();

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

	//Compacts the valid points from the point cloud buffer into the VBO.
	//Returns the number of valid elements
	int fillPointCloudVBO();

	static void glutIdle();
	static void glutDisplay();
	static void glutKeyboard(unsigned char key, int x, int y);
	static void glutReshape(int w, int h);
};

