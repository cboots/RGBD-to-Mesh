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

