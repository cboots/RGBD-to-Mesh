#include "LogDevice.h"


LogDevice::LogDevice(void)
{
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
	char *cstr = new char[logStr.length() + 1];
	strcpy(cstr, logStr.c_str());
	xml_document<> doc;    // character type defaults to char
	doc.parse<0>(cstr);    // 0 means default parse flags
	delete [] cstr;


	//Parse tree
	xml_node<char>* root = doc.first_node("device");
	if(root != 0)
	{
		//Load device
		xml_attribute<char>* xres = root->first_attribute("xresolution");
		xml_attribute<char>* yres = root->first_attribute("yresolution");
		if(xres != 0 && yres != 0){
			mXRes = atoi(xres->value);
			mYRes = atoi(yres->value);

			//Clear arrays
			mColorGuard.lock();
			mColorStreamFrames.clear();
			mColorGuard.unlock();
			mDepthGuard.lock();
			mDepthStreamFrames.clear();
			mDepthGuard.unlock();

			xml_node<char>* frame = root->first_node("frame");
			if(frame != 0){

				while(frame != 0){
					xml_attribute<char>* id = frame->first_attribute("id");
					xml_attribute<char>* colorTimestamp = frame->first_attribute("colorTimestamp");
					xml_attribute<char>* depthTimestamp = frame->first_attribute("depthTimestamp");

					if(id != 0){
						int frameId = atoi(id->value());
						if(colorTimestamp != 0)
						{
							timestamp time = atol(colorTimestamp->value());
							mColorGuard.lock();
							mColorStreamFrames.push_back(FrameMetaData(frameId, time));
							mColorGuard.unlock();
						}

						if(depthTimestamp != 0)
						{
							timestamp time = atol(depthTimestamp->value());
							mDepthGuard.lock();
							mDepthStreamFrames.push_back(FrameMetaData(frameId, time));
							mDepthGuard.unlock();
						}
					}
					frame->next_sibling("frame");
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
}

DeviceStatus LogDevice::connect(void)
{

	std::ostringstream logfilePath; 
	logfilePath << mDirectory << "\\log.xml" << endl;
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
	//TODO: implement
	return false;
}

bool LogDevice::hasColorStream()
{
	//TODO: implement
	return false;
}

bool LogDevice::createColorStream() 
{
	//TODO: Implement
	return false;
}

bool LogDevice::createDepthStream() 
{
	//TODO: Implement
	return false;
}

bool LogDevice::destroyColorStream()  
{
	return false;
}

bool LogDevice::destroyDepthStream()
{
	return false;
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
	return false;
}

bool LogDevice::isColorStreamValid()
{
	return false;
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