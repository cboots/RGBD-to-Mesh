#include "RGBDFrame.h"



namespace rgbd
{
	namespace framework
	{
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
			//NULL Pointers
			mDepthData = DPixelArray();
			mColorData = ColorPixelArray();

			mDepthTime = 0;
			mColorTime = 0;
			mHasDepth = false;
			mHasColor = false;

		}


		void RGBDFrame::resetMetaData(void)
		{	
			mDepthTime = 0;
			mColorTime = 0;
			mHasDepth = false;
			mHasColor = false;
		}

		//forceAlloc is default false.
		void RGBDFrame::setResolution(int width, int height, bool forceAlloc /*=false*/)
		{
			if(width != mXRes || height != mYRes || forceAlloc)
			{
				if(width > 0 && height > 0){
					//Resolution changed, reallocate
					mXRes = width;
					mYRes = height;
					mDepthData = DPixelArray(new DPixel[width*height]);
					mColorData = ColorPixelArray(new ColorPixel[width*height]);
				}else{
					//width and height invalid. Clear
					mXRes = 0;
					mYRes = 0;
					//Null pointers
					mDepthData = DPixelArray();
					mColorData = ColorPixelArray();
				}
			}
		}

		void RGBDFrame::clearDepthImage(void)
		{
			DPixel clear = {0};
			if(mDepthData != NULL){
				for(int y = 0; y < mYRes; y++)
				{
					for(int x = 0; x < mXRes; x++)
					{
						mDepthData[getLinearIndex(x,y)] = clear;
					}
				}
			}
		}

		void RGBDFrame::clearColorImage(void)
		{
			ColorPixel clear = {0,0,0};

			if(mColorData != NULL){
				for(int y = 0; y < mYRes; y++)
				{
					for(int x = 0; x < mXRes; x++)
					{
						mColorData[getLinearIndex(x,y)] = clear;
					}
				}
			}
		}

	}
}