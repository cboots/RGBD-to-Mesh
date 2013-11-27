#pragma once
#include "RGBDDevice.h"
#include <string>
#include <queue>
#include <boost/thread.hpp>
#include "FileUtils.h"
#include <sstream>

using namespace std;


class FrameLogger : public RGBDDevice::NewRGBDFrameListener
{
private:
	FrameLogger( const FrameLogger& other ); // non construction-copyable
	FrameLogger& operator=(const FrameLogger&);//Make not copiable

protected:
	string mOutputDirectory;
	queue<RGBDFramePtr> mFrameQueue;
	volatile bool mIsRecording;
	RGBDDevice* mDevice;

	boost::thread mLoggerThread;
	boost::mutex mQueueGuard;

	void record(string outputDirectory);
public:
	FrameLogger(void);
	~FrameLogger(void);


	//Cannot change directory during recording
	bool setOutputDirectory(string dir) 
	{
		if(mIsRecording)
			return false;
		mOutputDirectory = dir; 
		return true;
	}

	string getOutputDirectory() {return mOutputDirectory;}

	//Create the output directory. Returns false if directory could not be created or is not empty
	bool makeOutputDirectory();

	//Returns true if successfully created
	bool startRecording(RGBDDevice* device);
	void stopRecording();
	bool isRecording() {return mIsRecording;}


	void onNewRGBDFrame(RGBDFramePtr frame) override;

};

