#pragma once
#include "rgbddevice.h"
#include "RGBDFrameFactory.h"
#include <string>
#include "FileUtils.h"
#include <ostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <mutex>
#include "rapidxml\rapidxml.hpp"
#include <chrono>
#include <boost/date_time.hpp>
#include <stdlib.h>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace rapidxml;

struct FrameMetaData{
	int id;
	timestamp time;
	FrameMetaData(int id, timestamp time){
		this->id = id;
		this->time = time;
	}
};

class LogDevice :
	public RGBDDevice
{
protected:
	RGBDFrameFactory mFrameFactory;
	string mDirectory;

	int mXRes,mYRes;
	vector<FrameMetaData> mColorStreamFrames;
	vector<FrameMetaData> mDepthStreamFrames;

	//Stream management
	bool mSyncDepthAndColor;
	bool mLoopStreams;
	timestamp mStartTime;
	boost::posix_time::ptime mPlaybackStartTime;
	volatile bool mColorStreaming;
	volatile bool mDepthStreaming;
	volatile int mColorInd;
	volatile int mDepthInd;

	
	std::thread mColorThread;
	std::thread mDepthThread;
	std::mutex mColorGuard;
	std::mutex mDepthGuard;

	RGBDFramePtr mRGBDFrameSynced;
	std::mutex mFrameGuard;



	void loadColorFrame(string sourceDir, FrameMetaData data, RGBDFramePtr frameOut);
	void loadDepthFrame(string sourceDir, FrameMetaData data, RGBDFramePtr frameOut);
	void loadLog(string logFile);
	void streamColor();
	void streamDepth();
public:
	LogDevice(void);
	~LogDevice(void);

	DeviceStatus initialize(void)  override;//Initialize 
	DeviceStatus connect(void)	   override;//Connect to any device
	DeviceStatus disconnect(void)  override;//Disconnect from current device
	DeviceStatus shutdown(void) override;
	
	void restartStreams();

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

	inline void setLoopStreams(bool loop) {mLoopStreams = loop;}
	inline bool getLoopStreams(){return mLoopStreams;}

	
	inline virtual bool getSyncColorAndDepth() override {return mSyncDepthAndColor;}
	inline virtual bool setSyncColorAndDepth(bool sync) override { mSyncDepthAndColor = sync; return true;}
};

