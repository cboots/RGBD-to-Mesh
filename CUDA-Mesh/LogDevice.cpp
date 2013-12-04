#include "LogDevice.h"


LogDevice::LogDevice(void)
{
	mDirectory = "";

	mXRes = 0;
	mYRes = 0;

	//Stream management
	mSyncDepthAndColor = false;
	mLoopStreams = false;
	mStartTime = 0; 
	mColorStreaming = false;
	mDepthStreaming = false;
	mColorInd = 0;
	mDepthInd = 0;

	mRGBDFrameSynced = NULL;
	mPlaybackSpeed = 1.0;

}


LogDevice::~LogDevice(void)
{
}



DeviceStatus LogDevice::initialize(void)
{

	return DEVICESTATUS_OK;	
}

void LogDevice::loadLog(string logFile)
{
	string logStr = loadTextFile(logFile);

	//Need to copy string to parse it
	char *cstr = new char[logStr.length() + 1]();
	logStr.copy(cstr, logStr.length());
	xml_document<> doc;    // character type defaults to char
	doc.parse<0>(cstr);    // 0 means default parse flags


	//Parse tree
	xml_node<char>* root = doc.first_node("device");
	if(root != 0)
	{
		//Load device
		xml_attribute<char>* xres = root->first_attribute("xresolution");
		xml_attribute<char>* yres = root->first_attribute("yresolution");
		if(xres != 0 && yres != 0){
			mXRes = atoi(xres->value());
			mYRes = atoi(yres->value());

			//Clear arrays
			mColorGuard.lock();
			mColorStreamFrames.clear();
			mColorGuard.unlock();
			mDepthGuard.lock();
			mDepthStreamFrames.clear();
			mDepthGuard.unlock();

			xml_node<char>* frame = root->first_node("frame");
			if(frame != 0){
				timestamp depthStartTime = 0;
				timestamp colorStartTime = 0;
				while(frame != 0){
					xml_attribute<char>* id = frame->first_attribute("id");
					xml_attribute<char>* colorTimestamp = frame->first_attribute("colorTimestamp");
					xml_attribute<char>* depthTimestamp = frame->first_attribute("depthTimestamp");

					if(id != 0){
						int frameId = atoi(id->value());
						if(colorTimestamp != 0)
						{

							timestamp time = boost::lexical_cast<timestamp>(colorTimestamp->value());
							if(colorStartTime == 0)
								colorStartTime = time;

							mColorGuard.lock();
							mColorStreamFrames.push_back(FrameMetaData(frameId, time));
							mColorGuard.unlock();
						}

						if(depthTimestamp != 0)
						{
							timestamp time = boost::lexical_cast<timestamp>(depthTimestamp->value());
							if(depthStartTime == 0)
								depthStartTime = time;

							mDepthGuard.lock();
							mDepthStreamFrames.push_back(FrameMetaData(frameId, time));
							mDepthGuard.unlock();
						}
					}
					frame = frame->next_sibling("frame");
				}

				mStartTime = min(depthStartTime, colorStartTime);

			}else{
				onMessage("Empty Log File\n");
			}
		}else{
			onMessage("Missing Resolution Attributes\n");
		}
	}else{
		onMessage("Invalid Log File\n");
	}

	//Free memory
	delete [] cstr;
}

DeviceStatus LogDevice::connect(void)
{

	std::ostringstream logfilePath; 
	logfilePath << mDirectory << "\\log.xml";
	if(fileExists(logfilePath.str()))
	{

		//Load log.
		loadLog(logfilePath.str());

		onConnect();
		return DEVICESTATUS_OK;
	}else{
		std::ostringstream out; 
		out << "Couldn't open log file" << endl;
		onMessage(out.str());

		return DEVICESTATUS_NO_DEVICE;
	}

}

DeviceStatus LogDevice::disconnect(void)
{
	onDisconnect();
	return DEVICESTATUS_OK;
}

DeviceStatus LogDevice::shutdown(void)
{
	return DEVICESTATUS_OK;
}

bool LogDevice::hasDepthStream()
{
	return mDepthStreaming;
}

bool LogDevice::hasColorStream()
{
	return mColorStreaming;
}


void LogDevice::bufferColor()
{
	RGBDFramePtr localFrame = NULL;
	int bufferPos = 0;

	while(mColorStreaming){
		if(mColorStreamBuffer.size() < MAX_BUFFER_FRAMES)
		{
			//More room in buffer to fill
			//Check if stream has pending frames
			if(bufferPos < mColorStreamFrames.size())
			{
				FrameMetaData frame = mColorStreamFrames[bufferPos];

				localFrame = mFrameFactory.getRGBDFrame(mXRes, mYRes);
				loadColorFrame(mDirectory, frame, localFrame);
				
				//Buffer the frame
				mColorGuard.lock();
				mColorStreamBuffer.push(BufferFrame(frame.id, localFrame));
				localFrame = NULL;//Pass to buffer
				mColorGuard.unlock();

				bufferPos++;
			}

			//Loop buffer if turned on 
			if(mLoopStreams && mColorStreamFrames.size() <= bufferPos) 
			{
				bufferPos = 0;
			}
		}
		boost::this_thread::sleep_for(boost::chrono::milliseconds(1));//1ms resolution should be fine

	}

}


void LogDevice::bufferDepth()
{
	RGBDFramePtr localFrame = NULL;
	int bufferPos = 0;

	while(mDepthStreaming){
		if(mDepthStreamBuffer.size() < MAX_BUFFER_FRAMES)
		{
			//More room in buffer to fill
			//Check if stream has pending frames
			if(bufferPos < mDepthStreamFrames.size())
			{
				FrameMetaData frame = mDepthStreamFrames[bufferPos];

				localFrame = mFrameFactory.getRGBDFrame(mXRes, mYRes);
				loadDepthFrame(mDirectory, frame, localFrame);
				
				//Buffer the frame
				mDepthGuard.lock();
				mDepthStreamBuffer.push(BufferFrame(frame.id, localFrame));
				localFrame = NULL;//Pass to buffer
				mDepthGuard.unlock();

				bufferPos++;
			}

			//Loop buffer if turned on 
			if(mLoopStreams && mDepthStreamFrames.size() <= bufferPos) 
			{
				bufferPos = 0;
			}
		}
		boost::this_thread::sleep_for(boost::chrono::milliseconds(1));//1ms resolution should be fine

	}

}


void LogDevice::dispatchEvents()
{
	RGBDFramePtr localFrame = NULL;
	while(mColorStreaming || mDepthStreaming)
	{
		//Tick thread time
		boost::posix_time::ptime now  = boost::posix_time::microsec_clock::local_time();
		boost::posix_time::time_duration duration = (now - mPlaybackStartTime);
		timestamp durationUS = duration.total_microseconds()*mPlaybackSpeed;//Time in microseconds. Normalized to playback time

		//=======Check depth stream for update========
		if(mDepthStreaming){
			if(mDepthInd < mDepthStreamFrames.size()){
				FrameMetaData frame = mDepthStreamFrames[mDepthInd];
				//If we've passed frame time
				if(frame.time - mStartTime <= durationUS)
				{

					//TODO: Do buffer read here.
					//If buffer read success, mDepthInd++;
					


				}
			}else{
				//Reached end of stream, restart
				if(mLoopStreams){
					restartStreams();
					localFrame = NULL;//clear frame (avoids wrapped synchronization)
				}
			}
		}

		//=======Check color stream for update========
		if(mColorStreaming){
			if(mColorInd < mColorStreamFrames.size()){
				FrameMetaData frame = mColorStreamFrames[mColorInd];
				//If we've passed frame time
				if(frame.time - mStartTime <= durationUS)
				{

					//TODO: Do buffer read here.
					//If buffer read success, mColorInd++;



				}
			}else{
				//Reached end of stream, restart
				if(mLoopStreams){
					restartStreams();
					localFrame = NULL;//clear frame (avoids wrapped synchronization)
				}
			}
		}

		//Sleep thread
		boost::this_thread::sleep_for(boost::chrono::milliseconds(1));//1ms resolution should be fine
	}
}

void LogDevice::streamColor()
{
	while(mColorStreaming){
		boost::posix_time::ptime now  = boost::posix_time::microsec_clock::local_time();
		boost::posix_time::time_duration duration = (now - mPlaybackStartTime);
		timestamp durationUS = duration.total_microseconds()*mPlaybackSpeed;
		//Color
		mColorGuard.lock();
		if(mColorInd < mColorStreamFrames.size())
		{
			//TODO: Buffering
			FrameMetaData frame = mColorStreamFrames[mColorInd];
			//If we've passed frame time
			if(frame.time - mStartTime <= durationUS)
			{
				mColorInd++;
				mColorGuard.unlock();
				RGBDFramePtr localFrame = mFrameFactory.getRGBDFrame(mXRes, mYRes);
				loadColorFrame(mDirectory, frame, localFrame);

				//Check if send
				if(!mSyncDepthAndColor)
				{
					onNewRGBDFrame(localFrame);
					localFrame = NULL;
				}else{
					//Sync it
					mFrameGuard.lock();
					if(mRGBDFrameSynced == NULL)
					{
						//FIRST POST!!!
						mRGBDFrameSynced = localFrame;
					}else{
						//SECOND!!
						//Send it
						mRGBDFrameSynced->setColorArray(localFrame->getColorArray());
						mRGBDFrameSynced->setHasColor(true);
						mRGBDFrameSynced->setColorTimestamp(localFrame->getColorTimestamp());

						onNewRGBDFrame(mRGBDFrameSynced);
						mRGBDFrameSynced = NULL;
					}
					//Unlock scoped
					mFrameGuard.unlock();
				}
			}else{
				mColorGuard.unlock();//ALWAYS UNLOCK YOUR MUTEX!!!
			}
			//TODO: Syncing

		}else{
			//Reached end of stream
			if(mLoopStreams)
				restartStreams();

			mColorGuard.unlock();
		}
		//Sleep thread
		boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
	}
}


void LogDevice::streamDepth()
{
	boost::thread eventDispatch;
	while(mDepthStreaming){
		boost::posix_time::ptime now  = boost::posix_time::microsec_clock::local_time();
		boost::posix_time::time_duration duration = (now - mPlaybackStartTime);
		timestamp durationUS = duration.total_microseconds()*mPlaybackSpeed;
		//Depth
		mDepthGuard.lock();
		if(mDepthInd < mDepthStreamFrames.size())
		{
			//TODO: Buffering
			FrameMetaData frame = mDepthStreamFrames[mDepthInd];
			if(frame.time - mStartTime <= durationUS)
			{
				mDepthInd++;
				mDepthGuard.unlock();
				RGBDFramePtr localFrame = mFrameFactory.getRGBDFrame(mXRes, mYRes);
				loadDepthFrame(mDirectory, frame, localFrame);
				//Check if send
				if(!mSyncDepthAndColor)
				{
					onNewRGBDFrame(localFrame);
					localFrame = NULL;
				}else{
					//Sync it
					mFrameGuard.lock();
					if(mRGBDFrameSynced == NULL)
					{
						//FIRST POST!!!
						mRGBDFrameSynced = localFrame;
					}else{
						//SECOND!!
						//Send it
						mRGBDFrameSynced->setDepthArray(localFrame->getDepthArray());
						mRGBDFrameSynced->setHasDepth(true);
						mRGBDFrameSynced->setDepthTimestamp(localFrame->getDepthTimestamp());

						onNewRGBDFrame(mRGBDFrameSynced);
						mRGBDFrameSynced = NULL;
					}
					//Unlock scoped
					mFrameGuard.unlock();
				}
			}else{
				mDepthGuard.unlock();//ALWAYS UNLOCK YOUR MUTEX!!!
			}
			//TODO: Syncing
		}else{
			//Reached end of stream
			if(mLoopStreams)
				restartStreams();

			mDepthGuard.unlock();
		}
		//Sleep thread
		boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
	}
}

bool LogDevice::createColorStream() 
{
	if(mColorStreamFrames.size() > 0){
		mColorStreaming = true;

		//Start stream thread
		mColorThread = boost::thread(&LogDevice::streamColor, this);

		restartStreams();
	}
	return mColorStreaming;
}

bool LogDevice::createDepthStream() 
{

	if(mDepthStreamFrames.size() > 0){
		mDepthStreaming = true;

		//Start stream thread
		mDepthThread = boost::thread(&LogDevice::streamDepth, this);

		restartStreams();
	}
	return mDepthStreaming;
}

bool LogDevice::destroyColorStream()  
{
	mColorStreaming = false;
	return true;
}

bool LogDevice::destroyDepthStream()
{
	mDepthStreaming = false;
	return true;
}


void LogDevice::restartStreams()
{
	//Do not lock, will deadlock threads
	mColorInd = 0;
	mDepthInd = 0;
	mPlaybackStartTime  = boost::posix_time::microsec_clock::local_time();
}

int LogDevice::getDepthResolutionX()
{
	return mXRes;
}
int LogDevice::getDepthResolutionY()
{
	return mYRes;
}

int LogDevice::getColorResolutionX()
{
	return mXRes;
}

int LogDevice::getColorResolutionY()
{
	return mYRes;
}

bool LogDevice::isDepthStreamValid() 
{
	return mDepthStreaming;
}

bool LogDevice::isColorStreamValid()
{
	return mColorStreaming;
}

void LogDevice::loadColorFrame(string sourceDir, FrameMetaData data, RGBDFramePtr frameOut)
{
	std::ostringstream out; 
	out << sourceDir << "\\" << data.id;
	frameOut->setColorTimestamp(data.time);
	loadColorImageFromFile(out.str(), frameOut);
}

void LogDevice::loadDepthFrame(string sourceDir, FrameMetaData data, RGBDFramePtr frameOut)
{
	std::ostringstream out; 
	out << sourceDir << "\\" << data.id;
	frameOut->setDepthTimestamp(data.time);
	loadDepthImageFromFile(out.str(), frameOut);
}

void LogDevice::setPlaybackSpeed(double speed) 
{
	if(speed>0.0) {
		//Grab current time scaled by old playback
		boost::posix_time::ptime now  = boost::posix_time::microsec_clock::local_time();
		boost::posix_time::time_duration duration = (now - mPlaybackStartTime);
		timestamp durationUS = duration.total_microseconds()*mPlaybackSpeed;

		//Change speed
		mPlaybackSpeed = speed;

		if(mColorStreaming || mDepthStreaming)
		{
			//If playback ongoing, reset start time to simulate seamless speed change
			double usOffset = durationUS;
			usOffset /= mPlaybackSpeed;
			mPlaybackStartTime = now - boost::posix_time::microseconds(usOffset);//Move playback to scaled time
		}
	}
}