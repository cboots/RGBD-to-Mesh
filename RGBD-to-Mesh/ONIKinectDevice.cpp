#include "ONIKinectDevice.h"


ONIKinectDevice::ONIKinectDevice(void)
{
}


ONIKinectDevice::~ONIKinectDevice(void)
{
}


DeviceStatus ONIKinectDevice::initialize(void)
{

	Status rc = OpenNI::initialize();
	if (rc != STATUS_OK)
	{
		std::ostringstream out; 
		out << "Initialize failed\n" << OpenNI::getExtendedError() << "\n";
		onMessage(out.str());
		return DEVICESTATUS_ERROR;
	}else{
		onMessage("OpenNI device initialized\n");
		OpenNI::addDeviceConnectedListener(this);
		OpenNI::addDeviceDisconnectedListener(this);
		OpenNI::addDeviceStateChangedListener(this);
		return DEVICESTATUS_OK;
	}
}

DeviceStatus ONIKinectDevice::connect(void)
{
	Status rc = mDevice.open(ANY_DEVICE);
	if (rc != STATUS_OK)
	{
		std::ostringstream out; 
		out << "Couldn't open device\n" << OpenNI::getExtendedError() << "\n";
		onMessage(out.str());
		return DEVICESTATUS_ERROR;
	}else{
		onConnect();
		return DEVICESTATUS_OK;
	}
}

DeviceStatus ONIKinectDevice::disconnect(void)
{
	mDevice.close();
	onDisconnect();
	return DEVICESTATUS_OK;
}



DeviceStatus ONIKinectDevice::shutdown(void)
{
	OpenNI::shutdown();
	onMessage("OpenNI Shutdown\n");
	return DEVICESTATUS_OK;
}


bool ONIKinectDevice::hasDepthStream()
{
	return (mDevice.getSensorInfo(SENSOR_DEPTH) != NULL);
}

bool ONIKinectDevice::hasColorStream()
{
	return (mDevice.getSensorInfo(SENSOR_COLOR) != NULL);
}

bool ONIKinectDevice::createColorStream() 
{
	Status rc;
	if (mDevice.getSensorInfo(SENSOR_COLOR) != NULL)
	{
		rc = mColorStream.create(mDevice, SENSOR_COLOR);
		if (rc != STATUS_OK)
		{

			std::ostringstream out; 
			out << "Couldn't create color stream\n\n" << OpenNI::getExtendedError() << "\n";
			onMessage(out.str());
			return false;
		}
	}
	rc = mColorStream.start();
	if (rc != STATUS_OK)
	{
		std::ostringstream out; 
		out << "Couldn't start the color stream\n\n" << OpenNI::getExtendedError() << "\n";
		onMessage(out.str());
		return false;
	}

	newColorFrameListener = NewColorFrameListener(this);
	mColorStream.addNewFrameListener(&newColorFrameListener);

	return true;
}

bool ONIKinectDevice::createDepthStream() 
{

	Status rc;
	if (mDevice.getSensorInfo(SENSOR_DEPTH) != NULL)
	{
		rc = mDepthStream.create(mDevice, SENSOR_DEPTH);
		if (rc != STATUS_OK)
		{

			std::ostringstream out; 
			out << "Couldn't create depth stream\n\n" << OpenNI::getExtendedError() << "\n";
			onMessage(out.str());
			return false;
		}
	}
	rc = mDepthStream.start();
	if (rc != STATUS_OK)
	{
		std::ostringstream out; 
		out << "Couldn't start the depth stream\n\n" << OpenNI::getExtendedError() << "\n";
		onMessage(out.str());
		return false;
	}

	newDepthFrameListener = NewDepthFrameListener(this);
	mDepthStream.addNewFrameListener(&newDepthFrameListener);

	return true;
}

bool ONIKinectDevice::destroyColorStream()  
{
	mColorStream.stop();
	mColorStream.destroy();
	return true;
}

bool ONIKinectDevice::destroyDepthStream()
{
	mDepthStream.stop();
	mDepthStream.destroy();
	return true;
}



void ONIKinectDevice::onDeviceStateChanged(const DeviceInfo* pInfo, DeviceState state)
{
	std::ostringstream out; 
	out << "Device \"" << pInfo->getUri() << "\" error state changed to " << state << "\n";
	onMessage(out.str());
}

void ONIKinectDevice::onDeviceConnected(const DeviceInfo* pInfo)
{
	onConnect();
	std::ostringstream out; 
	out << "Device \"" << pInfo->getUri() << "\" connected\n";
	onMessage(out.str());
}

void ONIKinectDevice::onDeviceDisconnected(const DeviceInfo* pInfo)
{
	onDisconnect();
	std::ostringstream out; 
	out << "Device \"" << pInfo->getUri() << "\" disconnected\n";
	onMessage(out.str());
}


void ONIKinectDevice::onNewDepthFrame(VideoFrameRef frame)
{
	printf("[%08llu] Depth Frame\n", (long long)frame.getTimestamp());

	//Make sure frame is in right format
	if(frame.getVideoMode().getPixelFormat() == PIXEL_FORMAT_DEPTH_1_MM || 
		frame.getVideoMode().getPixelFormat() == PIXEL_FORMAT_DEPTH_100_UM )
	{
		int width = frame.getVideoMode().getResolutionX();
		int height = frame.getVideoMode().getResolutionY();

		//Initialize frame if not initialized
		if(mRGBDFrameDepth == NULL)
		{
			mRGBDFrameDepth = mFrameFactory.getRGBDFrame(width,height);
		}

		if(width == mRGBDFrameDepth->getXRes()  && height == mRGBDFrameDepth->getYRes())
		{
			//Data array valid. Fill it
			//TODO: Use more efficient method of transfering memory. (like memcopy, or plain linear indexing?)
			DepthPixelArray data = mRGBDFrameDepth->getDepthArray();
			//TODO: Enable cropping

			const openni::DepthPixel* pDepth = (const openni::DepthPixel*)frame.getData();
			for(int y = 0; y < height; y++)
			{
				for(int x = 0; x < width; x++)
				{
					int ind = mRGBDFrameDepth->getLinearIndex(x,y);
					data[ind].depth = pDepth[ind];
				}
			}

			mRGBDFrameDepth->setHasDepth(true);
			//Check if send
			if(!mSyncDepthAndColor || mRGBDFrameDepth->hasColor())
			{
				//Send it
				onNewRGBDFrame(mRGBDFrameDepth);
				mRGBDFrameDepth = NULL;
			}

		}else{
			//Size error
			onMessage("Error: depth and color frames don't match in size\n");	
		}

	}else{
		//Format error
		onMessage("Error: depth format incorrect\n");	
	}
}

void ONIKinectDevice::onNewColorFrame(VideoFrameRef frame)
{
	printf("[%08llu] Color Frame\n", (long long)frame.getTimestamp());

	//Make sure frame is in right format
	if(frame.getVideoMode().getPixelFormat() == PIXEL_FORMAT_RGB888)
	{
		int width = frame.getVideoMode().getResolutionX();
		int height = frame.getVideoMode().getResolutionY();

		//Initialize frame if not initialized
		if(mRGBDFrameColor == NULL)
		{
			mRGBDFrameColor = mFrameFactory.getRGBDFrame(width,height);
		}

		if(width == mRGBDFrameColor->getXRes()  && height == mRGBDFrameColor->getYRes())
		{
			//Data array valid. Fill it
			//TODO: Use more efficient method of transfering memory. (like memcopy, or plain linear indexing?)
			ColorPixelArray data = mRGBDFrameColor->getColorArray();
			//TODO: Enable cropping
			
			const openni::RGB888Pixel* pImage = (const openni::RGB888Pixel*)frame.getData();
			for(int y = 0; y < height; y++)
			{
				for(int x = 0; x < width; x++)
				{
					int ind = mRGBDFrameColor->getLinearIndex(x,y);
					data[ind].r = pImage[ind].r;
					data[ind].g = pImage[ind].g;
					data[ind].b = pImage[ind].b;
				}
			}

			mRGBDFrameColor->setHasColor(true);
			//Check if send
			if(!mSyncDepthAndColor || mRGBDFrameColor->hasDepth())
			{
				//Send it
				onNewRGBDFrame(mRGBDFrameColor);
				mRGBDFrameColor = NULL;
			}

		}else{
			//Size error
			onMessage("Error: depth and color frames don't match in size\n");	
		}

	}else{
		//Format error
		onMessage("Error: depth format incorrect\n");	
	}
}



bool ONIKinectDevice::setImageRegistrationMode(RGBDImageRegistrationMode mode)
{
	switch(mode)
	{
	case RGBDImageRegistrationMode::REGISTRATION_DEPTH_TO_COLOR:
		mDevice.setImageRegistrationMode(ImageRegistrationMode::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
		break;
	default:
		mDevice.setImageRegistrationMode(ImageRegistrationMode::IMAGE_REGISTRATION_OFF);
		break;
	}

	return true;
}

RGBDImageRegistrationMode ONIKinectDevice::getImageRegistrationMode(void)
{
	ImageRegistrationMode mode = mDevice.getImageRegistrationMode();
	switch(mode)
	{
	case ImageRegistrationMode::IMAGE_REGISTRATION_DEPTH_TO_COLOR:
		return RGBDImageRegistrationMode::REGISTRATION_DEPTH_TO_COLOR;
	default:
		return RGBDImageRegistrationMode::REGISTRATION_OFF;
	}
}