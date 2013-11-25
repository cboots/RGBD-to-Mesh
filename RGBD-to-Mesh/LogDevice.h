#pragma once
#include "rgbddevice.h"
#include "RGBDFrameFactory.h"
#include <string>
#include "FileUtils.h"
#include <ostream>
#include <fstream>
#include <sstream>

using namespace std;

class LogDevice :
	public RGBDDevice
{
protected:
	RGBDFrameFactory mFrameFactory;
	string mDirectory;

	int mXRes,mYRes;
public:
	LogDevice(void);
	~LogDevice(void);
	
	DeviceStatus initialize(void)  override;//Initialize 
	DeviceStatus connect(void)	   override;//Connect to any device
	DeviceStatus disconnect(void)  override;//Disconnect from current device
	DeviceStatus shutdown(void) override;

	void setSourceDirectory(string dir) { mDirectory = dir;}
	string setSourceDirectory(){return mDirectory;}

	bool hasDepthStream() override;
	bool hasColorStream() override;

	bool createColorStream() override;
	bool createDepthStream() override;

	bool destroyColorStream()  override;
	bool destroyDepthStream()  override;
	
	bool setImageRegistrationMode(RGBDImageRegistrationMode) override;
	RGBDImageRegistrationMode getImageRegistrationMode(void) override;

	int getDepthResolutionX() override;
	int getDepthResolutionY() override;
	int getColorResolutionX() override;
	int getColorResolutionY() override;
	bool isDepthStreamValid() override;
	bool isColorStreamValid() override;

};

