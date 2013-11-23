#pragma once
#include <OpenNI.h>
#include "rgbddevice.h"
#include <iostream>
#include <string>
#include <sstream>
#include "RGBDFrameFactory.h"

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
	RGBDFrameFactory mFrameFactory;

	//RGBDFramePtr mRGBDFrameDepth;
	//RGBDFramePtr mRGBDFrameColor;
	bool mSyncDepthAndColor;
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
	
	bool setImageRegistrationMode(RGBDImageRegistrationMode) override;
	RGBDImageRegistrationMode getImageRegistrationMode(void) override;

	//Override to implement
	inline virtual bool getSyncColorAndDepth() override {return mSyncDepthAndColor;}
	inline virtual bool setSyncColorAndDepth(bool sync) override { mSyncDepthAndColor = sync; return true;}


	//Event handlers
	void onDeviceStateChanged(const DeviceInfo* pInfo, DeviceState state);
	void onDeviceConnected(const DeviceInfo* pInfo);
	void onDeviceDisconnected(const DeviceInfo* pInfo);

	virtual void onNewDepthFrame(VideoFrameRef frame);
	virtual void onNewColorFrame(VideoFrameRef frame);
	
	int getDepthResolutionX() override;
	int getDepthResolutionY() override;
	int getColorResolutionX() override;
	int getColorResolutionY() override;
	bool isDepthStreamValid() override;
	bool isColorStreamValid() override;

};

