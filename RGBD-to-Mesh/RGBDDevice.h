#pragma once
#include "RGBDFrame.h"
#include <string>
#include <vector>
#include <algorithm>

enum ImageRegistrationMode
{
	REGISTRATION_OFF = 0,
	REGISTRATION_DEPTH_TO_COLOR = 1
};

class RGBDDevice
{
	//******************Events and listener declarations*****************
public:
	class NewRGBDFrameListener{
	public :
		virtual void onNewRGBDFrame(RGBDFramePtr frame) = 0;
	};

	class DeviceConnectedListener{
	public :
		virtual void onDeviceConnected() = 0;
	};

	class DeviceDisconnectedListener{
	public :
		virtual void onDeviceDisconnected() = 0;
	};

	class DeviceMessageListener{
	public :
		virtual void onMessage(std::string msg) = 0;
	};

protected:
	std::vector<NewRGBDFrameListener*> mNewRGBDFrameListeners;
	std::vector<DeviceConnectedListener*> mDeviceConnectedListeners;
	std::vector<DeviceDisconnectedListener*> mDeviceDisconnectedListeners;
	std::vector<DeviceMessageListener*> mDeviceMessageListeners;

public:
	virtual ~RGBDDevice(void){};

	virtual void start() = 0;

	//Override to enable streams
	virtual bool hasDepthStream() {return false;}
	virtual bool hasColorStream() {return false;}

	//Return true if successful (false usually means this device doesn't support the functionality)
	virtual bool createColorStream() { return false;}
	//Return true if successful (false usually means this device doesn't support the functionality)
	virtual bool createDepthStream() { return false;}
	//Return true if successful (false usually means this device doesn't support the functionality)
	virtual bool setImageRegistrationMode(ImageRegistrationMode mode) {return false;}

	virtual ImageRegistrationMode getImageRegistrationMode(ImageRegistrationMode mode) {return REGISTRATION_OFF;}

	void addNewRGBDFrameListener(NewRGBDFrameListener* listener);
	void addDeviceConnectedListener(DeviceConnectedListener* listener);
	void addDeviceDisconnectedListener(DeviceDisconnectedListener* listener);
	void addDeviceMessageListener(DeviceMessageListener* listener);

	void removeNewRGBDFrameListener(NewRGBDFrameListener* listener);
	void removeDeviceConnectedListener(DeviceConnectedListener* listener);
	void removeDeviceDisconnectedListener(DeviceDisconnectedListener* listener);
	void removeDeviceMessageListener(DeviceMessageListener* listener);

	void onNewRGBDFrame(RGBDFramePtr frame);
	void onConnect();
	void onDisconnect();
	void onMessage(std::string msg);
};
