#pragma once
#include "RGBDFrame.h"
#include <string>
#include <vector>
#include <algorithm>

enum RGBDImageRegistrationMode
{
	REGISTRATION_OFF = 0,
	REGISTRATION_DEPTH_TO_COLOR = 1
};

typedef enum
{
	DEVICESTATUS_OK = 0,
	DEVICESTATUS_ERROR = 1,
	DEVICESTATUS_NOT_IMPLEMENTED = 2,
	DEVICESTATUS_NOT_SUPPORTED = 3,
	DEVICESTATUS_BAD_PARAMETER = 4,
	DEVICESTATUS_OUT_OF_FLOW = 5,
	DEVICESTATUS_NO_DEVICE = 6,
	DEVICESTATUS_TIME_OUT = 102,
} DeviceStatus;

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
	bool mWaitForFullFrame;
public:
	virtual ~RGBDDevice(void){};

	//Purely virtual methods
	virtual DeviceStatus initialize(void) = 0;
	virtual DeviceStatus connect(void) = 0;
	virtual DeviceStatus disconnect(void) = 0;
	virtual DeviceStatus shutdown(void) = 0;

	//Override to enable streams
	inline virtual bool hasDepthStream() {return false;}
	inline virtual bool hasColorStream() {return false;}

	//Return true if successful (false usually means this device doesn't support the functionality)
	inline virtual bool createColorStream() { return false;}
	//Return true if successful (false usually means this device doesn't support the functionality)
	inline virtual bool createDepthStream() { return false;}

	
	//Return true if successful (false usually means this device doesn't support the functionality)
	inline virtual bool destroyColorStream() { return false;}
	//Return true if successful (false usually means this device doesn't support the functionality)
	inline virtual bool destroyDepthStream() { return false;}

	//Return true if successful (false usually means this device doesn't support the functionality)
	inline virtual bool setImageRegistrationMode(RGBDImageRegistrationMode) {return false;}

	inline virtual RGBDImageRegistrationMode getImageRegistrationMode(void) {return REGISTRATION_OFF;}

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
