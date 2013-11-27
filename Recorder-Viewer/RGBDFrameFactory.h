#pragma once
#include "RGBDFrame.h"

class RGBDFrameFactory
{
public:
	RGBDFrameFactory(void);
	~RGBDFrameFactory(void);


	//Returns a frame with no memory allocated.
	//The frame is guaranteed to be unused by other processes assuming other processes do not cast RGBDFramePtr to raw pointer
	//Metadata will be reset by the factory
	RGBDFramePtr getRGBDFrame(); 

	//Returns a frame with the specified resolution
	//The frame is guaranteed to be unused by other processes assuming other processes do not cast RGBDFramePtr to raw pointer
	//The memory is not garunteed to be clear. As an optimization, this has been left to the user code to specify
	RGBDFramePtr getRGBDFrame(int width, int height); 
};

