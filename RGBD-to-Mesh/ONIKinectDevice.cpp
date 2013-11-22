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
