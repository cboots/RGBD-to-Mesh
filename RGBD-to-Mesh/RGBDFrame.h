#pragma once
//Contains type definitions for RGBD frame storage and manipulation.

#include <stdint.h>
#include <memory>

using namespace std;

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
typedef shared_ptr<RGBDFrame> frame_ptr;
typedef shared_ptr<DepthPixel[]> depth_array_ptr;
typedef shared_ptr<ColorPixel[]> color_array_ptr;

class RGBDFrame
{
protected:
	int mXRes, mYRes;
	timestamp mDepthTime, mColorTime;
	bool mHasDepth, mHasColor;

	depth_array_ptr mDepthData;
	color_array_ptr mColorData;

public:
	RGBDFrame(void);
	~RGBDFrame(void);


	void setResolution(int width, int height);
	void clearDepthImage();
	void clearColorImage();
	

	inline color_array_ptr getColorArray()
	{
		return mColorData;
	}

	inline depth_array_ptr getDepthArray()
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

