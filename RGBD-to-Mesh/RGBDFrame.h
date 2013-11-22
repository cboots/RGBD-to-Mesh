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
	//As soon as this function returns, both color and depth arrays are initialized.
	//Depth and color arrays must be the same resolution.
	void setResolution(int width, int height);

	//Writes 0 to all elements of depth image
	void clearDepthImage(void);

	//Writes 0 to all elements of color image
	void clearColorImage(void);
	
	
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
};

