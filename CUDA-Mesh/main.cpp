#include "main.h"


using namespace std;

void pause();




class RGBDDeviceListener : public RGBDDevice::DeviceDisconnectedListener,
	public RGBDDevice::DeviceConnectedListener,
	public RGBDDevice::DeviceMessageListener
{
public:
	virtual void onMessage(string msg)
	{
		printf("%s", msg.c_str());
	}

	virtual void onDeviceConnected()
	{
		printf("Device connected\n");
	}

	virtual void onDeviceDisconnected()
	{
		printf("Device disconnected\n");
	}
};

class RGBDFrameListener : public RGBDDevice::NewRGBDFrameListener
{
public:
	void onNewRGBDFrame(RGBDFramePtr frame) override
	{
		//printf("C[%08llu]D[%08llu] New Frame Recieved: Has Color %d Has Depth %d\n", frame->getColorTimestamp(), frame->getDepthTimestamp(), frame->hasColor(), frame->hasDepth());
	}
};


const int testArraySizeX = 640;
const int testArraySizeY = 2;
float host_onesArray[testArraySizeY][testArraySizeX];


int main(int argc, char** argv)
{
	boost::filesystem::path full_path( boost::filesystem::current_path() );
	cout << full_path << endl;


	RGBDDevice* devicePtr;

	//Load source argument
	if(argc > 1)
	{
		devicePtr = new LogDevice();
		((LogDevice*)devicePtr)->setSourceDirectory(argv[1]);
		((LogDevice*)devicePtr)->setLoopStreams(true);

	}else{
		devicePtr = new ONIKinectDevice();
	}	

	RGBDDeviceListener deviceStateListener;
	RGBDFrameListener frameListener;
	devicePtr->addDeviceConnectedListener(&deviceStateListener);
	devicePtr->addDeviceDisconnectedListener(&deviceStateListener);
	devicePtr->addDeviceMessageListener(&deviceStateListener);
	devicePtr->addNewRGBDFrameListener(&frameListener);

	devicePtr->initialize();
	if(DEVICESTATUS_OK != devicePtr->connect())
	{
		printf("Could not connect to device\n");
		devicePtr->shutdown();
		pause();
		return 1;
	}

	if(!devicePtr->createDepthStream())
	{
		printf("Could not create depth stream\n");
		devicePtr->shutdown();
		pause();
		return 2;
	}

	if(!devicePtr->createColorStream())
	{
		printf("Could not create color stream\n");
		devicePtr->shutdown();
		pause();
		return 3;
	}

	printf("Streams created succesfully\n");


	devicePtr->setImageRegistrationMode(REGISTRATION_DEPTH_TO_COLOR);
	devicePtr->setSyncColorAndDepth(false);

	MeshViewer viewer(devicePtr, screenwidth, screenheight);

	DeviceStatus rc = viewer.init(argc, argv);
	if (rc != DEVICESTATUS_OK)
	{
		devicePtr->shutdown();
		pause();
		return 4;
	}


	viewer.run();
	pause();

	return 0;
}


void pause()
{
	cout << "Press enter to continue..." << endl;
	cin.ignore();
}