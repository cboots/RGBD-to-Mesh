#pragma once
class RGBDDevice
{
private:
	RGBDDevice(void);
public:
	virtual ~RGBDDevice(void) = 0;
};


//Even virtual destructors MUST exist
RGBDDevice::~RGBDDevice(void){ }

