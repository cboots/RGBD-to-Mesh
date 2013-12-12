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


int main(int argc, char** argv)
{
	boost::filesystem::path full_path( boost::filesystem::current_path() );
	cout << full_path << endl;

	LogDevice device;
	device.setSourceDirectory("logs\\flat-wall-synced");
	device.setLoopStreams(true);
	//ONIKinectDevice device;

	RGBDDeviceListener deviceStateListener;
	RGBDFrameListener frameListener;
	device.addDeviceConnectedListener(&deviceStateListener);
	device.addDeviceDisconnectedListener(&deviceStateListener);
	device.addDeviceMessageListener(&deviceStateListener);
	device.addNewRGBDFrameListener(&frameListener);

	device.initialize();
	if(DEVICESTATUS_OK != device.connect())
	{
		printf("Could not connect to device\n");
		device.shutdown();
		pause();
		return 1;
	}

	if(!device.createDepthStream())
	{
		printf("Could not create depth stream\n");
		device.shutdown();
		pause();
		return 2;
	}

	if(!device.createColorStream())
	{
		printf("Could not create color stream\n");
		device.shutdown();
		pause();
		return 3;
	}

	printf("Streams created succesfully\n");

	
	device.setImageRegistrationMode(REGISTRATION_DEPTH_TO_COLOR);
	device.setSyncColorAndDepth(false);

	MeshViewer viewer(&device, screenwidth, screenheight);

	DeviceStatus rc = viewer.init(argc, argv);
	if (rc != DEVICESTATUS_OK)
	{
		device.shutdown();
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