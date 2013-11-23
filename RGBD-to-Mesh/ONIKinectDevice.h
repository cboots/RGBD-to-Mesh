#pragma once
#include <OpenNI.h>
#include "rgbddevice.h"
#include <iostream>
#include <string>
#include <sstream>

using namespace std;
using namespace openni;



class ONIKinectDevice :
	public RGBDDevice,
	//Event listeners
	public OpenNI::DeviceConnectedListener,
	public OpenNI::DeviceDisconnectedListener,
	public OpenNI::DeviceStateChangedListener
{
protected:

	class NewDepthFrameListener:public VideoStream::NewFrameListener{

	private:
		VideoFrameRef m_frame;
		ONIKinectDevice* outer;
	public:
		NewDepthFrameListener() { outer = NULL;}
		NewDepthFrameListener(ONIKinectDevice* o)
		{
			outer = o;
		}

		void onNewFrame(VideoStream& stream)
		{
			stream.readFrame(&m_frame);
			if(outer != NULL)
				outer->onNewDepthFrame(m_frame);
		}
	} newDepthFrameListener;

	class NewColorFrameListener:public VideoStream::NewFrameListener{

	private:
		VideoFrameRef m_frame;
		ONIKinectDevice* outer;
	public:
		NewColorFrameListener() { outer = NULL;}
		NewColorFrameListener(ONIKinectDevice* o)
		{
			outer = o;
		}

		void onNewFrame(VideoStream& stream)
		{
			stream.readFrame(&m_frame);

			if(outer != NULL)
				outer->onNewColorFrame(m_frame);
		}
	} newColorFrameListener;


	Device mDevice;
	VideoStream mDepthStream;
	VideoStream mColorStream;

	RGBDFramePtr mRGBDFrame;
public:
	ONIKinectDevice(void);
	~ONIKinectDevice(void);

	DeviceStatus initialize(void)  override;//Initialize 
	DeviceStatus connect(void)	   override;//Connect to any device
	DeviceStatus disconnect(void)  override;//Disconnect from current device
	DeviceStatus shutdown(void) override;

	bool hasDepthStream() override;
	bool hasColorStream() override;

	bool createColorStream() override;
	bool createDepthStream() override;

	bool destroyColorStream()  override;
	bool destroyDepthStream()  override;

	//Event handlers
	void onDeviceStateChanged(const DeviceInfo* pInfo, DeviceState state);
	void onDeviceConnected(const DeviceInfo* pInfo);
	void onDeviceDisconnected(const DeviceInfo* pInfo);

	virtual void onNewDepthFrame(VideoFrameRef frame);
	virtual void onNewColorFrame(VideoFrameRef frame);

};

