#include "RGBDFrameFactory.h"



namespace rgbd
{
	namespace framework
	{

		RGBDFrameFactory::RGBDFrameFactory(void)
		{
		}


		RGBDFrameFactory::~RGBDFrameFactory(void)
		{
		}



		//Returns a frame with no memory allocated.
		//The frame is guaranteed to be unused by other processes assuming other processes do not cast RGBDFramePtr to raw pointer
		//Metadata will be reset by the factory
		RGBDFramePtr RGBDFrameFactory::getRGBDFrame()
		{
			return getRGBDFrame(0,0);
		}

		//Returns a frame with the specified resolution
		//The frame is guaranteed to be unused by other processes assuming other processes do not cast RGBDFramePtr to raw pointer
		//The memory is not garunteed to be clear. As an optimization, this has been left to the user code to specify
		RGBDFramePtr RGBDFrameFactory::getRGBDFrame(int width, int height)
		{
			//TODO: Impelement frame recycling for memory efficiency
			return RGBDFramePtr(new RGBDFrame(width, height));
		}

	}
}