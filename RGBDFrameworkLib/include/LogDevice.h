#pragma once
#include "rgbddevice.h"
#include "RGBDFrameFactory.h"
#include <string>
#include "FileUtils.h"
#include <ostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <boost/thread.hpp>
#include "../rapidxml/rapidxml.hpp"
#include <boost/date_time.hpp>
#include <stdlib.h>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace rapidxml;



namespace rgbd
{
	namespace framework
	{
		//TODO: Make setable parameter
#define MAX_BUFFER_FRAMES 150 //5 seconds of data


		struct FrameMetaData{
			int id;
			timestamp time;
			COMPRESSION_METHOD compressionMode;
			FrameMetaData(int id, timestamp time, COMPRESSION_METHOD compressionMode){
				this->id = id;
				this->time = time;
				this->compressionMode = compressionMode;
			}
		};

		struct SyncFrameMetaData{
			FrameMetaData depthData;
			FrameMetaData colorData;
		};

		struct BufferFrame{
			RGBDFramePtr frame;
			int id;

			BufferFrame(int id, RGBDFramePtr frame)
			{
				this->id = id;
				this->frame = frame;
			}
		};

		class LogDevice :
			public RGBDDevice
		{
		protected:
			//Provides new frames 
			RGBDFrameFactory mFrameFactory;
			//Source directory for log files
			string mDirectory;

			//Resolution of logged data
			int mXRes,mYRes;

			//Stream frames for various 
			vector<SyncFrameMetaData> mLogFrames;

			//Stream management
			bool mLoopStreams;
			timestamp mStartTime;
			boost::posix_time::ptime mPlaybackStartTime;
			volatile bool mColorStreaming;
			volatile bool mDepthStreaming;
			volatile int mColorInd;
			volatile int mDepthInd;


			//Stream frame buffers.
			//const int cBufferCapacity = 500;//Max number of frames to buffer per stream
			queue<BufferFrame> mColorStreamBuffer;
			queue<BufferFrame> mDepthStreamBuffer;

			//1.0 is normal, 0.5 is half speed, 2.0 is double speed, etc
			double mPlaybackSpeed;

			boost::thread mColorThread;
			boost::thread mDepthThread;
			boost::thread mEventThread;
			boost::mutex mLogGuard;

			RGBDFramePtr mRGBDFrameSynced;
			boost::mutex mFrameGuard;



			void loadColorFrame(string sourceDir, FrameMetaData data, RGBDFramePtr frameOut, COMPRESSION_METHOD colorCompressMode);
			void loadDepthFrame(string sourceDir, FrameMetaData data, RGBDFramePtr frameOut, COMPRESSION_METHOD depthCompressMode);
			void loadLog(string logFile);
			void bufferColor();
			void bufferDepth();
			void dispatchEvents();
			void insertColorFrameToSyncedFrames(FrameMetaData colorData);
		public:
			LogDevice(void);
			~LogDevice(void);

			DeviceStatus initialize(void)  override;//Initialize 
			DeviceStatus connect(void)	   override;//Connect to any device
			DeviceStatus disconnect(void)  override;//Disconnect from current device
			DeviceStatus shutdown(void) override;

			void restartPlayback();

			void setSourceDirectory(string dir) { mDirectory = dir;}
			string setSourceDirectory(){return mDirectory;}

			bool hasDepthStream() override;
			bool hasColorStream() override;

			bool createColorStream() override;
			bool createDepthStream() override;

			bool destroyColorStream()  override;
			bool destroyDepthStream()  override;

			int getDepthResolutionX() override;
			int getDepthResolutionY() override;
			int getColorResolutionX() override;
			int getColorResolutionY() override;
			bool isDepthStreamValid() override;
			bool isColorStreamValid() override;

			//If set to true, whenever either stream reaches the end of the loop the playback will restart from the beginning.
			inline void setLoopStreams(bool loop) {mLoopStreams = loop;}
			inline bool getLoopStreams(){return mLoopStreams;}

			bool getSyncColorAndDepth() override {return true;}
			bool setSyncColorAndDepth(bool sync) override { return false;}

			//Getter/Setter for playback speed
			//1.0 is normal, 0.5 is half speed, 2.0 is double speed, etc
			void setPlaybackSpeed(double speed);
			double getPlaybackSpeed(){return mPlaybackSpeed;}
		};

	}
}