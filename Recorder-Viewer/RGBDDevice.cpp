#include "RGBDDevice.h"

void RGBDDevice::addNewRGBDFrameListener(NewRGBDFrameListener* listener)
{
	mNewRGBDFrameListeners.push_back(listener);
}

void RGBDDevice::addDeviceConnectedListener(DeviceConnectedListener* listener)
{
	mDeviceConnectedListeners.push_back(listener);
}

void RGBDDevice::addDeviceDisconnectedListener(DeviceDisconnectedListener* listener)
{
	mDeviceDisconnectedListeners.push_back(listener);
}

void RGBDDevice::addDeviceMessageListener(DeviceMessageListener* listener)
{
	mDeviceMessageListeners.push_back(listener);
}

void RGBDDevice::removeNewRGBDFrameListener(NewRGBDFrameListener* listener)
{
	std::vector<NewRGBDFrameListener*>::iterator position = std::find(mNewRGBDFrameListeners.begin(), mNewRGBDFrameListeners.end(), listener);

	if (position != mNewRGBDFrameListeners.end()) // == vector.end() means the element was not found
		mNewRGBDFrameListeners.erase(position);

}

void RGBDDevice::removeDeviceConnectedListener(DeviceConnectedListener* listener)
{
	std::vector<DeviceConnectedListener*>::iterator position = std::find(mDeviceConnectedListeners.begin(), mDeviceConnectedListeners.end(), listener);

	if (position != mDeviceConnectedListeners.end()) // == vector.end() means the element was not found
		mDeviceConnectedListeners.erase(position);

}

void RGBDDevice::removeDeviceDisconnectedListener(DeviceDisconnectedListener* listener)
{
	std::vector<DeviceDisconnectedListener*>::iterator position = std::find(mDeviceDisconnectedListeners.begin(), mDeviceDisconnectedListeners.end(), listener);

	if (position != mDeviceDisconnectedListeners.end()) // == vector.end() means the element was not found
		mDeviceDisconnectedListeners.erase(position);

}

void RGBDDevice::removeDeviceMessageListener(DeviceMessageListener* listener)
{
	std::vector<DeviceMessageListener*>::iterator position = std::find(mDeviceMessageListeners.begin(), mDeviceMessageListeners.end(), listener);

	if (position != mDeviceMessageListeners.end()) // == vector.end() means the element was not found
		mDeviceMessageListeners.erase(position);

}

void RGBDDevice::onNewRGBDFrame(RGBDFramePtr frame)
{
	for(std::vector<NewRGBDFrameListener*>::iterator it = mNewRGBDFrameListeners.begin(); it != mNewRGBDFrameListeners.end(); ++it) {
		boost::thread eventDispatch = boost::thread(&NewRGBDFrameListener::onNewRGBDFrame, (*it), frame);
		eventDispatch.detach();
		//(*it)->onNewRGBDFrame(frame);
	}
}

void RGBDDevice::onConnect()
{
	for(std::vector<DeviceConnectedListener*>::iterator it = mDeviceConnectedListeners.begin(); it != mDeviceConnectedListeners.end(); ++it) {
		boost::thread eventDispatch = boost::thread(&DeviceConnectedListener::onDeviceConnected, (*it));
		eventDispatch.detach();
		//(*it)->onDeviceConnected();
	}
}

void RGBDDevice::onDisconnect()
{
	for(std::vector<DeviceDisconnectedListener*>::iterator it = mDeviceDisconnectedListeners.begin(); it != mDeviceDisconnectedListeners.end(); ++it) {
		boost::thread eventDispatch = boost::thread(&DeviceDisconnectedListener::onDeviceDisconnected, (*it));
		eventDispatch.detach();
		//(*it)->onDeviceDisconnected();
	}
}

void RGBDDevice::onMessage(std::string msg)
{
	for(std::vector<DeviceMessageListener*>::iterator it = mDeviceMessageListeners.begin(); it != mDeviceMessageListeners.end(); ++it) {
		boost::thread eventDispatch = boost::thread(&DeviceMessageListener::onMessage, (*it), msg);
		eventDispatch.detach();
		//(*it)->onMessage(msg);
	}
}
