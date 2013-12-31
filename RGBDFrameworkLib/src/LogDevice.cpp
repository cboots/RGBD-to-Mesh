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

		void LogDevice::insertColorFrameToSyncedFrames(FrameMetaData colorData)
		{
			//Assumes depth data is sorted.
			//Starts search at frame id and does gradient descent to find minimum dt
			mFrameGuard.lock();

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

			mFrameGuard.unlock();
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
							insertColorFrameToSyncedFrames(*it);
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
						loadColorFrame(mDirectory, frame, localFrame, frame.compressionMode);

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
						loadDepthFrame(mDirectory, frame, localFrame, frame.compressionMode);

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
			RGBDFramePtr localDepthFrame = NULL;
			RGBDFramePtr localColorFrame = NULL;
			RGBDFramePtr localSyncFrame = NULL;

			while(mColorStreaming || mDepthStreaming)
			{
				//Tick thread time
				boost::posix_time::ptime now  = boost::posix_time::microsec_clock::local_time();
				boost::posix_time::time_duration duration = (now - mPlaybackStartTime);
				timestamp durationUS = (timestamp) (duration.total_microseconds()*mPlaybackSpeed);//Time in microseconds. Normalized to playback time

				//=======Check depth stream for update========
				if(mDepthStreaming){
					localDepthFrame = NULL;//Clear
					if(mDepthInd < mDepthStreamFrames.size()){
						FrameMetaData frame = mDepthStreamFrames[mDepthInd];
						//If we've passed frame time
						if(frame.time - mStartTime <= durationUS)
						{
							mDepthGuard.lock();
							while(mDepthStreamBuffer.size() > 0){
								BufferFrame bufFrame = mDepthStreamBuffer.front();
								mDepthStreamBuffer.pop();
								if(frame.id == bufFrame.id)
								{
									//Found the correct position in the buffer, pull the frame
									localDepthFrame = bufFrame.frame;
									break;
								}
							}
							mDepthGuard.unlock();

							if(localDepthFrame != NULL)
							{
								//Found the right frame in the buffer, advance
								mDepthInd++;
							}
						}
					}else{
						//Reached end of stream, restart
						if(mLoopStreams){
							restartPlayback();
							localColorFrame = NULL;
							localDepthFrame = NULL;
						}
					}
				}

				//=======Check color stream for update========
				if(mColorStreaming){
					localColorFrame = NULL;//Clear
					if(mColorInd < mColorStreamFrames.size()){
						FrameMetaData frame = mColorStreamFrames[mColorInd];
						//If we've passed frame time
						if(frame.time - mStartTime <= durationUS)
						{

							mColorGuard.lock();
							while(mColorStreamBuffer.size() > 0){
								BufferFrame bufFrame = mColorStreamBuffer.front();
								mColorStreamBuffer.pop();
								if(frame.id == bufFrame.id)
								{
									//Found the correct position in the buffer, pull the frame
									localColorFrame = bufFrame.frame;
									break;
								}
								//Buffer miss
							}
							mColorGuard.unlock();

							if(localColorFrame != NULL)
							{
								//Found the right frame in the buffer, advance
								mColorInd++;
							}


						}
					}else{
						//Reached end of stream, restart
						if(mLoopStreams){
							restartPlayback();
							localColorFrame = NULL;
							localDepthFrame = NULL;
						}
					}
				}

				//Dispatch events if apprpriate
				//All synchronization logic handled below
				if(mSyncDepthAndColor)
				{
					//Synchronized
					//Check if anything new
					if(localColorFrame != NULL)
					{
						//New color data from stream
						if(localDepthFrame != NULL)
						{
							//Also new depth data. Both color and depth present
							if(localSyncFrame != NULL)
							{
								//has a frame already. complete the frame and send it
								if(localSyncFrame->hasColor())
								{
									//Synch packet and send
									localSyncFrame->overwriteDepthData(localDepthFrame);
									onNewRGBDFrame(localSyncFrame);

									//queue other packet
									localSyncFrame = localColorFrame;
								}else{
									//has depth
									localSyncFrame->overwriteColorData(localColorFrame);
									onNewRGBDFrame(localSyncFrame);

									//queue other packet
									localSyncFrame = localDepthFrame;
								}
							}else{
								localSyncFrame = localColorFrame;
								localSyncFrame->overwriteDepthData(localDepthFrame);
								onNewRGBDFrame(localSyncFrame);
								localSyncFrame = NULL;
							}

						}else{
							//New color data only
							if(localSyncFrame != NULL)
							{
								if(localSyncFrame->hasColor())
								{
									//Already has color data, push unsynchronized event
									onNewRGBDFrame(localSyncFrame);
									localSyncFrame = localColorFrame;
								}else{
									//has depth data, overwrite color
									localSyncFrame->overwriteColorData(localColorFrame);
									onNewRGBDFrame(localSyncFrame);
									localSyncFrame = NULL;
								}
							}else{
								//First one here
								localSyncFrame = localColorFrame;
							}

						}
					}else if(localDepthFrame != NULL){
						//New depth data only
						if(localSyncFrame != NULL)
						{
							if(localSyncFrame->hasDepth())
							{
								//already has depth data, push out unsynchronized event
								onNewRGBDFrame(localSyncFrame);
								localSyncFrame = localDepthFrame;
							}else{
								//Has color data, add depth
								localSyncFrame->overwriteDepthData(localDepthFrame);

								//Send synched event
								onNewRGBDFrame(localSyncFrame);
								localSyncFrame = NULL;
							}
						}else{
							//First one here
							localSyncFrame = localDepthFrame;
						}
					}

				}else{
					//Seperate dispatches
					if(localColorFrame != NULL)
					{
						onNewRGBDFrame(localColorFrame);
					}

					if(localDepthFrame != NULL)
					{
						onNewRGBDFrame(localDepthFrame);
					}
				}

				localDepthFrame = NULL;
				localColorFrame = NULL;
				//Sleep thread
				boost::this_thread::sleep_for(boost::chrono::milliseconds(1));//1ms resolution should be fine
			}
		}

		bool LogDevice::createColorStream() 
		{
			if(mColorStreamFrames.size() > 0){
				mColorStreaming = true;

				//Start stream thread
				mColorThread = boost::thread(&LogDevice::bufferColor, this);

				if(!mDepthStreaming)
				{
					//Event thread not started yet
					mEventThread = boost::thread(&LogDevice::dispatchEvents, this);
				}

				restartPlayback();
			}
			return mColorStreaming;
		}

		bool LogDevice::createDepthStream() 
		{

			if(mDepthStreamFrames.size() > 0){
				mDepthStreaming = true;

				//Start stream thread
				mDepthThread = boost::thread(&LogDevice::bufferDepth, this);

				if(!mColorStreaming)
				{
					//Event thread not started yet
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