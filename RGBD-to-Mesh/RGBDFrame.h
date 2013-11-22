#pragma once
//Contains type definitions for RGBD frame storage and manipulation.

#include <stdint.h>
#include "boost/shared_ptr.hpp"
#include "boost/shared_array.hpp"

//Pixel data format
typedef struct
{
	/* Red value of this pixel. */
	uint8_t r;
	/* Green value of this pixel. */
	uint8_t g;
	/* Blue value of this pixel. */
	uint8_t b;
} ColorPixel;

typedef struct
{
	/* Depth value of this pixel */
	uint16_t depth;
} DepthPixel;

typedef uint64_t timestamp;


//Forward declaration
class RGBDFrame;
typedef boost::shared_ptr<RGBDFrame> RGBDFramePtr;
typedef boost::shared_array<DepthPixel> DepthPixelArray;
typedef boost::shared_array<ColorPixel> ColorPixelArray;

class RGBDFrame
{
protected:
	int mXRes, mYRes;
	timestamp mDepthTime, mColorTime;
	bool mHasDepth, mHasColor;

	DepthPixelArray mDepthData;
	ColorPixelArray mColorData;


	void init();
public:
	RGBDFrame(void);
	RGBDFrame(int width, int height);
	~RGBDFrame(void);

	//Set Resolution must be called before accessing data arrays.
	//This method will reallocate a new array if the size has changed.
	//Data is not presevered during resizing.
	//
	//As soon as this function returns, both color and depth arrays have been allocated.
	//Depth and color arrays must be the same resolution.
	//You can pass an optional variable that will forec reallocation of the memory array
	//
	//Passing invalid width or height (either <= 0) will result in the image memory being nullified.
	//Other processes with DepthPixelArray or ColorPixelArray references may still use the data safely, but it will be deleted when the reference goes out of scope.
	void setResolution(int width, int height, bool forceAlloc = false);

	//Writes 0 to all elements of depth image
	void clearDepthImage(void);

	//Writes 0 to all elements of color image
	void clearColorImage(void);
	
	//Resets all metadata parameters. Does not affect resolution or image data
	void resetMetaData(void);
	
	//Returns managed pointer to color data array.
	//If setResolution has not been called yet, this function cannot be used.
	//This array can be accessed regardless of the state of hasColor.
	//Arrays are stored in row major order.
	inline ColorPixelArray getColorArray()
	{
		return mColorData;
	}

	//Returns managed pointer to depth data array.
	//If setResolution has not been called yet, this function cannot be used.
	//This array can be accessed regardless of the state of hasDepth.
	//Arrays are stored in row major order.
	inline DepthPixelArray getDepthArray()
	{
		return mDepthData;
	}


	inline void setColorPixel(int x, int y, ColorPixel pixel)
	{
		if(x >= 0 && y >= 0 && x < mXRes && y < mYRes)
		{
			mColorData[getLinearIndex(x,y)] = pixel;
		}
	}

	
	inline void setDepthPixel(int x, int y, DepthPixel pixel)
	{
		if(x >= 0 && y >= 0 && x < mXRes && y < mYRes)
		{
			mDepthData[getLinearIndex(x,y)] = pixel;
		}
	}

	inline DepthPixel getDepthPixel(int x, int y)
	{
		if(x >= 0 && y >= 0 && x < mXRes && y < mYRes)
		{
			return	mDepthData[getLinearIndex(x,y)];
		}
		
		return DepthPixel();
	}

	
	inline ColorPixel getColorPixel(int x, int y)
	{
		if(x >= 0 && y >= 0 && x < mXRes && y < mYRes)
		{
			return	mColorData[getLinearIndex(x,y)];
		}
		
		return ColorPixel();
	}

	inline void setHasDepth(bool hasDepth)
	{
		mHasDepth = hasDepth;
	}

	inline void setHasColor(bool hasColor)
	{
		mHasDepth = hasColor;
	}

	inline bool hasDepth()
	{
		return mHasDepth;
	}

	inline bool hasColor()
	{
		return mHasColor;
	}
	
	inline int getXRes()
	{
		return mXRes;
	}

	inline int getYRes()
	{
		return mYRes;
	}

	inline void setDepthTimestamp(timestamp depthTime)
	{
		mDepthTime = depthTime;
	}

	inline void setColorTimestamp(timestamp colorTime)
	{
		mColorTime = colorTime;
	}

	inline timestamp getDepthTimestamp()
	{
		return mDepthTime;
	}

	inline timestamp getColorTimestamp()
	{
		return mColorTime;
	}

	inline int getLinearIndex(int x, int y)
	{
		return x+y*mXRes;
	}

};

