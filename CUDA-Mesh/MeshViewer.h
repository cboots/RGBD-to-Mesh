#pragma once


#include <iostream>
#include "RGBDDevice.h"
#include "RGBDFrame.h"
#include "RGBDFrameFactory.h"
#include "FileUtils.h"
#include "FrameLogger.h"
#include "device_structs.h"
#include "Device.h"
#include "Utility.h"

using namespace glm;

enum DisplayModes
{
	DISPLAY_MODE_OVERLAY,
	DISPLAY_MODE_DEPTH,
	DISPLAY_MODE_IMAGE,
	DISPLAY_MODE_POINT_CLOUD
};

namespace quad_attributes {
	enum {
		POSITION,
		TEXCOORD
	};
}



class MeshViewer : public RGBDDevice::NewRGBDFrameListener
{
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
	GLuint colorTexture;//Image size
	GLuint depthTexture;//Image size
	GLuint pointCloudTexture;//Screen size


private:
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

	//PBOs
	GLuint imagePBO;

	void initShader();
	void initQuad();
	void initPBO();

	void drawQuad(GLuint prog, float xNDC, float yNDC, float widthScale, float heightScale, GLuint texture);

	//Draws depth image buffer to the texture.
	//Texture width and height must match the resolution of the depth image.
	//Returns false if width or height does not match, true otherwise
	bool drawDepthImageBufferToTexture(GLuint texture);

	//Draws color image buffer to the texture.
	//Texture width and height must match the resolution of the color image.
	//Returns false if width or height does not match, true otherwise
	bool drawColorImageBufferToTexture(GLuint texture);

	static void glutIdle();
	static void glutDisplay();
	static void glutKeyboard(unsigned char key, int x, int y);
	static void glutReshape(int w, int h);
};

