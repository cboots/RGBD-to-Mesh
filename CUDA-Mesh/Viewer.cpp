/*****************************************************************************
*                                                                            *
*  OpenNI 2.x Alpha                                                          *
*  Copyright (C) 2012 PrimeSense Ltd.                                        *
*                                                                            *
*  This file is part of OpenNI.                                              *
*                                                                            *
*  Licensed under the Apache License, Version 2.0 (the "License");           *
*  you may not use this file except in compliance with the License.          *
*  You may obtain a copy of the License at                                   *
*                                                                            *
*      http://www.apache.org/licenses/LICENSE-2.0                            *
*                                                                            *
*  Unless required by applicable law or agreed to in writing, software       *
*  distributed under the License is distributed on an "AS IS" BASIS,         *
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
*  See the License for the specific language governing permissions and       *
*  limitations under the License.                                            *
*                                                                            *
*****************************************************************************/
// Undeprecate CRT functions
#ifndef _CRT_SECURE_NO_DEPRECATE 
#define _CRT_SECURE_NO_DEPRECATE 1
#endif

#include "Viewer.h"

#if (ONI_PLATFORM == ONI_PLATFORM_MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "OniSampleUtilities.h"

#define GL_WIN_SIZE_X	1280
#define GL_WIN_SIZE_Y	1024
#define TEXTURE_SIZE	512

#define DEFAULT_DISPLAY_MODE	DISPLAY_MODE_DEPTH

#define MIN_NUM_CHUNKS(data_size, chunk_size)	((((data_size)-1) / (chunk_size) + 1))
#define MIN_CHUNKS_SIZE(data_size, chunk_size)	(MIN_NUM_CHUNKS(data_size, chunk_size) * (chunk_size))

SampleViewer* SampleViewer::ms_self = NULL;

void SampleViewer::glutIdle()
{
	glutPostRedisplay();
}
void SampleViewer::glutDisplay()
{
	SampleViewer::ms_self->display();
}
void SampleViewer::glutKeyboard(unsigned char key, int x, int y)
{
	SampleViewer::ms_self->onKey(key, x, y);
}






SampleViewer::SampleViewer(const char* strSampleName, RGBDDevice* device) :
	m_eViewState(DEFAULT_DISPLAY_MODE), m_pTexMap(NULL)

{
	mDevice = device;
	ms_self = this;
	strncpy(m_strSampleName, strSampleName, ONI_MAX_STR);
}
SampleViewer::~SampleViewer()
{
	delete[] m_pTexMap;

	ms_self = NULL;
}

openni::Status SampleViewer::init(int argc, char **argv)
{

	if (mDevice->isDepthStreamValid() && mDevice->isColorStreamValid())
	{

		int depthWidth = mDevice->getDepthResolutionX();
		int depthHeight = mDevice->getDepthResolutionY();
		int colorWidth = mDevice->getColorResolutionX();
		int colorHeight = mDevice->getColorResolutionY();

		if (depthWidth == colorWidth &&
			depthHeight == colorHeight)
		{
			m_width = depthWidth;
			m_height = depthHeight;

			printf("Color and depth same resolution: D: %dx%d, C: %dx%d\n",
				depthWidth, depthHeight,
				colorWidth, colorHeight);
		}
		else
		{
			printf("Error - expect color and depth to be in same resolution: D: %dx%d, C: %dx%d\n",
				depthWidth, depthHeight,
				colorWidth, colorHeight);
			return openni::STATUS_ERROR;
		}
	}
	else if (mDevice->isDepthStreamValid())
	{
		m_width = mDevice->getDepthResolutionX();
		m_height = mDevice->getDepthResolutionY();
	}
	else if (mDevice->isColorStreamValid())
	{
		m_width = mDevice->getColorResolutionX();
		m_height = mDevice->getColorResolutionY();
	}
	else
	{
		printf("Error - expects at least one of the streams to be valid...\n");
		return openni::STATUS_ERROR;
	}

	// Texture map init
	m_nTexMapX = MIN_CHUNKS_SIZE(m_width, TEXTURE_SIZE);
	m_nTexMapY = MIN_CHUNKS_SIZE(m_height, TEXTURE_SIZE);
	m_pTexMap = new openni::RGB888Pixel[m_nTexMapX * m_nTexMapY];

	//Register frame listener
	mDevice->addNewRGBDFrameListener(this);
	return initOpenGL(argc, argv);

}
openni::Status SampleViewer::run()	//Does not return
{
	glutMainLoop();

	return openni::STATUS_OK;
}
void SampleViewer::display()
{
	ColorPixelArray localColorArray = mColorArray;
	DPixelArray localDepthArray = mDepthArray;

	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, GL_WIN_SIZE_X, GL_WIN_SIZE_Y, 0, -1.0, 1.0);



	memset(m_pTexMap, 0, m_nTexMapX*m_nTexMapY*sizeof(openni::RGB888Pixel));

	// check if we need to draw image frame to texture
	if ((m_eViewState == DISPLAY_MODE_OVERLAY ||
		m_eViewState == DISPLAY_MODE_IMAGE) && localColorArray != NULL)
	{
		const ColorPixelArray colorArray = localColorArray;
		ColorPixel* pImageRow = colorArray.get();
		openni::RGB888Pixel* pTexRow = m_pTexMap;
		int rowsize = m_width;
		for (int y = 0; y < m_height; ++y)
		{

			openni::RGB888Pixel* pTex = pTexRow;
			ColorPixel* pImage = pImageRow;
			for (int x = 0; x <  m_width; ++x, ++pTex, ++pImage)
			{
				ColorPixel color = (*pImage);
				pTex->r  =  color.r;
				pTex->g  =  color.g;
				pTex->b  =  color.b;
			}
			pTexRow += m_nTexMapX;
			pImageRow += rowsize;
		}
	}

	if ((m_eViewState == DISPLAY_MODE_OVERLAY ||
		m_eViewState == DISPLAY_MODE_DEPTH) && localDepthArray != NULL)
	{
		const DPixelArray depthArray = localDepthArray;
		DPixel* pDepthRow = depthArray.get();
		openni::RGB888Pixel* pTexRow = m_pTexMap;
		int rowsize = m_width;
		for (int y = 0; y < m_height; ++y)
		{
			openni::RGB888Pixel* pTex = pTexRow;
			DPixel* pDepth = pDepthRow;
			for (int x = 0; x <  m_width; ++x, ++pTex, ++pDepth)
			{
				int depth = (*pDepth).depth;
				if(depth != 0){
					uint8_t scaledDepth = 256-(depth>>6);
					pTex->r = scaledDepth;
					pTex->g = 0;//scaledDepth;
					pTex->b = 0;//scaledDepth;
				}
			}
			pTexRow += m_nTexMapX;
			pDepthRow += rowsize;
		}
	}

	glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_nTexMapX, m_nTexMapY, 0, GL_RGB, GL_UNSIGNED_BYTE, m_pTexMap);

	// Display the OpenGL texture map
	glColor4f(1,1,1,1);

	glBegin(GL_QUADS);

	int nXRes = m_width;
	int nYRes = m_height;

	// upper left
	glTexCoord2f(0, 0);
	glVertex2f(0, 0);
	// upper right
	glTexCoord2f((float)nXRes/(float)m_nTexMapX, 0);
	glVertex2f(GL_WIN_SIZE_X, 0);
	// bottom right
	glTexCoord2f((float)nXRes/(float)m_nTexMapX, (float)nYRes/(float)m_nTexMapY);
	glVertex2f(GL_WIN_SIZE_X, GL_WIN_SIZE_Y);
	// bottom left
	glTexCoord2f(0, (float)nYRes/(float)m_nTexMapY);
	glVertex2f(0, GL_WIN_SIZE_Y);

	glEnd();

	// Swap the OpenGL display buffers
	glutSwapBuffers();

}


FrameLogger logger;
void SampleViewer::onKey(unsigned char key, int /*x*/, int /*y*/)
{
	float newPlayback = 1.0;
	switch (key)
	{
	case 27:
		mDevice->destroyColorStream();
		mDevice->destroyDepthStream();

		mDevice->disconnect();
		mDevice->shutdown();
		exit (1);
	case '1':
		m_eViewState = DISPLAY_MODE_OVERLAY;
		mDevice->setImageRegistrationMode(RGBDImageRegistrationMode::REGISTRATION_DEPTH_TO_COLOR);
		mDevice->setSyncColorAndDepth(true);
		break;
	case '2':
		m_eViewState = DISPLAY_MODE_DEPTH;
		mDevice->setImageRegistrationMode(RGBDImageRegistrationMode::REGISTRATION_OFF);
		mDevice->setSyncColorAndDepth(false);
		break;
	case '3':
		m_eViewState = DISPLAY_MODE_IMAGE;
		mDevice->setImageRegistrationMode(RGBDImageRegistrationMode::REGISTRATION_OFF);
		mDevice->setSyncColorAndDepth(false);
		break;
	case 'r':
		//Start recording
		if(!logger.setOutputDirectory("logs/recording"))
			cout<<"Could not set output directory"<<endl;

		if(!logger.startRecording(mDevice))
			cout << "Could not start recording" <<endl;
		else
			cout<<"Recording to :" << logger.getOutputDirectory() << endl;
		break;
	case 's':
		//Stop recording
		logger.stopRecording();
		cout<<"Recording stopped" <<endl;
		break;
	case 'p':
		((LogDevice*) mDevice)->restartPlayback();
		break;
	case '=':
		newPlayback = ((LogDevice*) mDevice)->getPlaybackSpeed()+0.1;
		cout <<"Playback speed: " << newPlayback << endl;
		((LogDevice*) mDevice)->setPlaybackSpeed(newPlayback);
		break;
	case '-':
		newPlayback = ((LogDevice*) mDevice)->getPlaybackSpeed()-0.1;
		cout <<"Playback speed: " << newPlayback << endl;
		((LogDevice*) mDevice)->setPlaybackSpeed(newPlayback);
		break;
	}

}

openni::Status SampleViewer::initOpenGL(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(GL_WIN_SIZE_X, GL_WIN_SIZE_Y);
	glutCreateWindow (m_strSampleName);
	// 	glutFullScreen();
	glutSetCursor(GLUT_CURSOR_NONE);

	initOpenGLHooks();

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);

	return openni::STATUS_OK;

}

void SampleViewer::initOpenGLHooks()
{
	glutKeyboardFunc(glutKeyboard);
	glutDisplayFunc(glutDisplay);
	glutIdleFunc(glutIdle);
}

void SampleViewer::onNewRGBDFrame(RGBDFramePtr frame)
{
	latestFrame = frame;
	if(latestFrame != NULL)
	{
		if(latestFrame->hasColor())
		{
			mColorArray = latestFrame->getColorArray();
		}

		if(latestFrame->hasDepth())
		{
			mDepthArray = latestFrame->getDepthArray();
		}
	}
}