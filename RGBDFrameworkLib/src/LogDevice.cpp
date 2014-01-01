#include "LogDevice.h"



namespace rgbd
{
	namespace framework
	{
		LogDevice::LogDevice(void)
		{
			mDirectory = "";

			mXRes = 0;
			mYRes = 0;

			//Stream management
			mLoopStreams = false;
			mStartTime = 0; 
			mColorStreaming = false;
			mDepthStreaming = false;
			mLogInd = 0;

			mPlaybackSpeed = 1.0;

			mBufferCapacity = 150;

		}


		LogDevice::~LogDevice(void)
		{
		}



		DeviceStatus LogDevice::initialize(void)
		{

			return DEVICESTATUS_OK;	
		}

		void LogDevice::insertColorFrameToSyncedFrames(FrameMetaData colorData)
		{
			//Assumes depth data is sorted.
			//Starts search at frame id and does gradient descent to find minimum dt
			int index = colorData.id - 1;//Ids a 1 based index
			std::vector<SyncFrameMetaData>::iterator it = mLogFrames.begin() + index;

			timestamp deltaT = colorData.time - it->depthData.time;//Current gradient

			//If not first element and previous element is closer, move down the list
			while(it != mLogFrames.begin() && (deltaT > colorData.time - (it-1)->depthData.time))
			{
				--it;
				deltaT = colorData.time - it->depthData.time;//Current gradient
			}

			//If not last element and next element is closer, move up
			while(it != mLogFrames.end() - 1 && (deltaT > colorData.time - (it+1)->depthData.time))
			{
				++it;
				deltaT = colorData.time - it->depthData.time;//Current gradient
			}

			//At correct insertion point. (minimum difference)
			it->colorData = colorData;

		}

		void LogDevice::loadLog(string logFile)
		{
			string logStr = loadTextFile(logFile);

			//Need to copy string to parse it
			char *cstr = new char[logStr.length() + 1]();
			logStr.copy(cstr, logStr.length());
			xml_document<> doc;    // character type defaults to char
			doc.parse<0>(cstr);    // 0 means default parse flags

			vector<FrameMetaData> colorFrames;

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

					//Clear array
					mLogGuard.lock();
					mLogFrames.clear();
					mLogGuard.unlock();

					xml_node<char>* frame = root->first_node("frame");
					if(frame != 0){
						timestamp depthStartTime = 0;
						timestamp colorStartTime = 0;
						while(frame != 0){
							xml_attribute<char>* id = frame->first_attribute("id");
							xml_attribute<char>* colorTimestamp = frame->first_attribute("colorTimestamp");
							xml_attribute<char>* depthTimestamp = frame->first_attribute("depthTimestamp");
							xml_attribute<char>* colorCompression = frame->first_attribute("colorCompression");
							xml_attribute<char>* depthCompression = frame->first_attribute("depthCompression");

							if(id != 0){
								int frameId = atoi(id->value());
								if(colorTimestamp != 0)
								{

									timestamp time = boost::lexical_cast<timestamp>(colorTimestamp->value());
									if(colorStartTime == 0)
										colorStartTime = time;
									COMPRESSION_METHOD compressionMethod = NO_COMPRESSION;
									if(colorCompression != 0)
									{
										char* compressionAlgId = colorCompression->value();
										compressionMethod = getCompressionMethodFromTag(string(compressionAlgId));
									}

									colorFrames.push_back(FrameMetaData(frameId, time, compressionMethod));
								}

								if(depthTimestamp != 0)
								{
									timestamp time = boost::lexical_cast<timestamp>(depthTimestamp->value());
									if(depthStartTime == 0)
										depthStartTime = time;

									COMPRESSION_METHOD compressionMethod = NO_COMPRESSION;
									if(depthCompression != 0)
									{
										char* compressionAlgId = depthCompression->value();
										compressionMethod = getCompressionMethodFromTag(string(compressionAlgId));
									}

									//Build master list from depth frames
									mLogGuard.lock();
									SyncFrameMetaData syncFrame;
									syncFrame.depthData = FrameMetaData(frameId, time, compressionMethod);
									mLogFrames.push_back(syncFrame);
									mLogGuard.unlock();
								}

							}
							frame = frame->next_sibling("frame");
						}

						mStartTime = min(depthStartTime, colorStartTime);

						//Merge color and depth streams by timestamp

						for(std::vector<FrameMetaData>::iterator it = colorFrames.begin(); it != colorFrames.end(); ++it)
						{
							mLogGuard.lock();
							insertColorFrameToSyncedFrames(*it);
							mLogGuard.unlock();
						}


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


		void LogDevice::bufferFrames()
		{
			RGBDFramePtr localFrame = NULL;
			mLogInd = 0;

			while(mColorStreaming || mDepthStreaming){
				if(mStreamBuffer.size() < mBufferCapacity)
				{
					//More room in buffer to fill
					//Check if stream has pending frames
					if(mLogInd < mLogFrames.size())
					{
						
						//Buffer the frame
						mBufferGuard.lock();
						SyncFrameMetaData frame = mLogFrames[mLogInd];

						localFrame = mFrameFactory.getRGBDFrame(mXRes, mYRes);
						if(mColorStreaming && frame.colorData.id > 0)
							loadColorFrame(mDirectory, frame.colorData, localFrame, frame.colorData.compressionMode);
						if(mDepthStreaming && frame.depthData.id > 0)
							loadDepthFrame(mDirectory, frame.depthData, localFrame, frame.depthData.compressionMode);

						mStreamBuffer.push(BufferFrame(frame.depthData.id, localFrame));
						localFrame = NULL;//Pass to buffer
						mBufferGuard.unlock();

						mLogInd++;
					}

					//Loop buffer if turned on 
					if(mLoopStreams && mLogFrames.size() <= mLogInd) 
					{
						mLogInd = 0;
					}
				}
				boost::this_thread::sleep_for(boost::chrono::milliseconds(10));//10ms resolution should be fine

			}

		}


		void LogDevice::dispatchEvents()
		{
			while(mColorStreaming || mDepthStreaming)
			{
				//Tick thread time
				boost::posix_time::ptime now  = boost::posix_time::microsec_clock::local_time();
				boost::posix_time::time_duration duration = (now - mPlaybackStartTime);
				timestamp currentPlaybackTimeUS = (timestamp) (duration.total_microseconds()*mPlaybackSpeed);//Time in microseconds. Normalized to playback time

				//Check for next time frame

				if(mStreamBuffer.size() > 0){
					mBufferGuard.lock();
					BufferFrame bufFrame = mStreamBuffer.front();
					timestamp nextTimeUS = bufFrame.time - mStartTime;


					mBufferGuard.unlock();
					if(nextTimeUS > 0)
					{
						if(nextTimeUS <= currentPlaybackTimeUS)
						{
							//Reached time to dispatch frame
							mBufferGuard.lock();
							mStreamBuffer.pop();
							mBufferGuard.unlock();
							onNewRGBDFrame(bufFrame.frame);
						}else{
							//Sleep until appropriate time
							boost::this_thread::sleep_for(boost::chrono::microseconds((timestamp) ((nextTimeUS - currentPlaybackTimeUS)/mPlaybackSpeed)));

						}
					}

				}else{
					boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
				}
			}
		}

		bool LogDevice::createColorStream() 
		{
			if(mLogFrames.size() > 0){

				mColorStreaming = true;
				if(!mDepthStreaming)
				{
					//Start threads
					mBufferThread = boost::thread(&LogDevice::bufferFrames, this);
					mEventThread = boost::thread(&LogDevice::dispatchEvents, this);
				}

				restartPlayback();
			}
			return mColorStreaming;
		}

		bool LogDevice::createDepthStream() 
		{

			if(mLogFrames.size() > 0){

				mDepthStreaming = true;
				if(!mColorStreaming)
				{
					//Start threads
					mBufferThread = boost::thread(&LogDevice::bufferFrames, this);
					mEventThread = boost::thread(&LogDevice::dispatchEvents, this);
				}

				restartPlayback();
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


		void LogDevice::restartPlayback()
		{
			//Do not lock, will deadlock threads
			mBufferGuard.lock();
			mLogInd = 0;
			mStreamBuffer = queue<BufferFrame>();
			mBufferGuard.unlock();//ALWAYS UNLOCK YOUR GORRAM MUTEX!
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

		void LogDevice::loadColorFrame(string sourceDir, FrameMetaData data, RGBDFramePtr frameOut, COMPRESSION_METHOD colorCompressMode)
		{
			std::ostringstream out; 
			out << sourceDir << "\\" << data.id;
			frameOut->setColorTimestamp(data.time);
			loadColorImageFromFile(out.str(), frameOut, colorCompressMode);
		}

		void LogDevice::loadDepthFrame(string sourceDir, FrameMetaData data, RGBDFramePtr frameOut, COMPRESSION_METHOD depthCompressMode)
		{
			std::ostringstream out; 
			out << sourceDir << "\\" << data.id;
			frameOut->setDepthTimestamp(data.time);
			loadDepthImageFromFile(out.str(), frameOut, depthCompressMode);
		}

		void LogDevice::setPlaybackSpeed(double speed) 
		{
			if(speed>0.0) {
				//Grab current time scaled by old playback
				boost::posix_time::ptime now  = boost::posix_time::microsec_clock::local_time();
				boost::posix_time::time_duration duration = (now - mPlaybackStartTime);
				timestamp durationUS = (timestamp) (duration.total_microseconds()*mPlaybackSpeed);

				//Change speed
				mPlaybackSpeed = speed;

				if(mColorStreaming || mDepthStreaming)
				{
					//If playback ongoing, reset start time to simulate seamless speed change
					double usOffset = (double) durationUS;
					usOffset /= mPlaybackSpeed;
					mPlaybackStartTime = now - boost::posix_time::microseconds((int64_t) usOffset);//Move playback to scaled time
				}
			}
		}



	}
}