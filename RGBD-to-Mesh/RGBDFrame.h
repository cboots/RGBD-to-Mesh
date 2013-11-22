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


//Forward declaration
class RGBDFrame;
typedef shared_ptr<RGBDFrame> frame_ptr;

class RGBDFrame
{

protected:
	DepthPixel* depthData;
	ColorPixel* colorData;

public:
	RGBDFrame(void);
	~RGBDFrame(void);

	bool hasDepth();
	bool hasColor();
	int getXRes();
	int getYRes();

	void setDepthData

};

