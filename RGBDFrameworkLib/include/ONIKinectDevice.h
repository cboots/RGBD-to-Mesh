#pragma once
#include <OpenNI.h>
#include "rgbddevice.h"
#include <iostream>
#include <string>
#include <sstream>
#include <boost/thread/mutex.hpp>
#include "RGBDFrameFactory.h"

using namespace std;
using namespace openni;



namespace rgbd
{
	namespace framework
	{

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

			//OpenNI Device
			Device mDevice;
			//Streams
			VideoStream mDepthStream;
			VideoStream mColorStream;
			//Factory for frame
			RGBDFrameFactory mFrameFactory;

			//Lock for synchronized frames
			boost::mutex frameGuard;
			RGBDFramePtr mRGBDFrameSynced;
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
			bool getSyncColorAndDepth() override {return mSyncDepthAndColor;}
			bool setSyncColorAndDepth(bool sync) override { mSyncDepthAndColor = sync; return true;}

			inline virtual Intrinsics getColorIntrinsics() 
				{return Intrinsics(526.37013657, 526.37013657, 313.68782938, 259.01834898);}
			inline virtual Intrinsics getDepthIntrinsics() 
				{return Intrinsics(585.05108211, 585.05108211, 315.83800193, 242.94140713);}

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

	}
}