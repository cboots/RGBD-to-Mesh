#include "LogDevice.h"


LogDevice::LogDevice(void)
{
}


LogDevice::~LogDevice(void)
{
}



DeviceStatus LogDevice::initialize(void)
{
	return DEVICESTATUS_OK;	
}

DeviceStatus LogDevice::connect(void)
{

	std::ostringstream logfilePath; 
	logfilePath << mDirectory << "\\log.xml" << endl;
	if(fileExists(logfilePath.str()))
	{
		onConnect();

		//Load log.

		return DEVICESTATUS_OK;
	}else{
		std::ostringstream out; 
		out << "Couldn't open log file" << endl;
		onMessage(out.str());

		return DEVICESTATUS_NO_DEVICE;
	}

}

DeviceStatus LogDevice::disconnect(void)
{
	onDisconnect();
	return DEVICESTATUS_OK;
}



DeviceStatus LogDevice::shutdown(void)
{
	return DEVICESTATUS_OK;
}


bool LogDevice::hasDepthStream()
{
	//TODO: implement
	return false;
}

bool LogDevice::hasColorStream()
{
	//TODO: implement
	return false;
}

bool LogDevice::createColorStream() 
{
	//TODO: Implement
	return false;
}

bool LogDevice::createDepthStream() 
{
	//TODO: Implement
	return false;
}

bool LogDevice::destroyColorStream()  
{
	return false;
}

bool LogDevice::destroyDepthStream()
{
	return false;
}


int LogDevice::getDepthResolutionX()
{
	return mXRes;
}
int LogDevice::getDepthResolutionY()
{
	return mYRes;
}

int LogDevice::getColorResolutionX()
{
	return mXRes;
}

int LogDevice::getColorResolutionY()
{
	return mYRes;
}


bool LogDevice::isDepthStreamValid() 
{
	return false;
}

bool LogDevice::isColorStreamValid()
{
	return false;
}