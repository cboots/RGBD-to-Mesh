#include "RGBDFrame.h"


RGBDFrame::RGBDFrame(void)
{
	init();
}

RGBDFrame::RGBDFrame(int width, int height)
{
	init();
	setResolution(width, height);
}


RGBDFrame::~RGBDFrame(void)
{
	//Clear arrays;
	mDepthData.reset();
	mColorData.reset();
}


void RGBDFrame::init(void)
{
	mXRes = 0;
	mYRes = 0;
	
	mDepthTime = 0;
	mColorTime = 0;
	mHasDepth = false;
	mHasColor = false;

}

void RGBDFrame::setResolution(int width, int height)
{
	if(width != mXRes || height != mYRes)
	{
		if(width > 0 && height > 0){
			//Resolution changed, reallocate
			mXRes = width;
			mYRes = height;
			mDepthData = DepthPixelArray(new DepthPixel[width*height]);
			mColorData = ColorPixelArray(new ColorPixel[width*height]);
		}
	}
}

void RGBDFrame::clearDepthImage(void)
{

	if(mDepthData != NULL){
		for(int y = 0; y < mYRes; y++)
		{
			for(int x = 0; x < mXRes; x++)
			{
				mDepthData[x+y*mXRes].depth = 0;
			}
		}
	}
}

void RGBDFrame::clearColorImage(void)
{

	if(mColorData != NULL){
		for(int y = 0; y < mYRes; y++)
		{
			for(int x = 0; x < mXRes; x++)
			{
				mColorData[x+y*mXRes].r = 0;
				mColorData[x+y*mXRes].g = 0;
				mColorData[x+y*mXRes].b = 0;
			}
		}
	}
}