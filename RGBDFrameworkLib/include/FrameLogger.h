#pragma once
#include "RGBDDevice.h"
#include <string>
#include <queue>
#include <boost/thread.hpp>
#include "FileUtils.h"
#include <sstream>


using namespace std;

namespace rgbd
{
	namespace framework
	{
		/*
		*	Class FrameLogger
		*	Tool for recording RGBDDevice streams. Can be used to asynchronously record any RGBDDevice's output frames
		*/
		class FrameLogger : public RGBDDevice::NewRGBDFrameListener
		{
		private:
			FrameLogger( const FrameLogger& other ); // non construction-copyable
			FrameLogger& operator=(const FrameLogger&);//Make not copiable

		protected:
			//Output directory to save log to
			string mOutputDirectory;

			//Buffer that allows log to record data without majorly affecting frame rate.
			//Event handler pushes frames to the queue
			queue<RGBDFramePtr> mFrameQueue;

			//Flag indicating if the log is recording
			volatile bool mIsRecording;

			//Device being recorded
			RGBDDevice* mDevice;

			//Thread for saving to file from buffer
			boost::thread mLoggerThread;

			//Mutex that should be locked whenever queue is being accessed.
			boost::mutex mQueueGuard;

			//Compression algorithm to use when saving color images
			COMPRESSION_METHOD mColorCompressionMethod;

			//Compression algorithm to use when saving depth images
			COMPRESSION_METHOD mDepthCompressionMethod;

			//This function will be run in mLoggerThread to save frames to output directory.
			void record(string outputDirectory);
		public:
			FrameLogger(void);
			~FrameLogger(void);


			//Sets the currect output directory for the logger
			//Cannot change directory during recording
			//Returns false if recording in progress and could not change directoy
			bool setOutputDirectory(string dir) 
			{
				if(mIsRecording)
					return false;
				mOutputDirectory = dir; 
				return true;
			}

			//Gets the current output directory.
			string getOutputDirectory() {return mOutputDirectory;}

			//Create the output directory. Returns false if directory could not be created or is not empty
			bool makeOutputDirectory();

			//Start recording the device
			//Adds a new frame listener to the device
			//Returns true if successfully created
			bool startRecording(RGBDDevice* device);

			//Stop recording the device. Has no effect if recording finished.
			void stopRecording();

			//Returns true if logger is recording a stream
			bool isRecording() {return mIsRecording;}

			//Event handler. Inherited from RGBDDevice::NewRGBDFrameListener
			void onNewRGBDFrame(RGBDFramePtr frame) override;

			inline void setColorCompressionMethod(COMPRESSION_METHOD method){ mColorCompressionMethod = method;}
			inline void setDepthCompressionMethod(COMPRESSION_METHOD method){ mDepthCompressionMethod = method;}
			inline COMPRESSION_METHOD getColorCompressionMethod(){return mColorCompressionMethod;}
			inline COMPRESSION_METHOD getDepthCompressionMethod(){return mDepthCompressionMethod;}

		};

	}
}