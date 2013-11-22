#pragma once
#include "rgbddevice.h"
class LogDevice :
	public RGBDDevice
{
public:
	LogDevice(void);
	~LogDevice(void);
};

