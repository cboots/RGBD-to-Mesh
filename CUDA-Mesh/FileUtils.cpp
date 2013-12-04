#include "FileUtils.h"

//Returns compression ratio for reference, or -1 if write failed
float saveToCompressedBinaryFile(string filename, char* uncompressed, int uncompressedSize, COMPRESSION_METHOD compressMode)
{
	char* compressed;
	int compressedSize;
	float ratio = -1.0f;
	//Try to open file
	ofstream file (filename, ios::out|ios::binary);
	if (file.is_open())
	{
		switch(compressMode)
		{
		case LZ4:
			//Allocate memory for compressed data
			compressed = new char[uncompressedSize];
			compressedSize = LZ4_compress_limitedOutput(uncompressed, compressed, uncompressedSize, uncompressedSize);
			ratio = float(compressedSize)/float(uncompressedSize);
			file.write(compressed, compressedSize);
			delete compressed;

			break;
		case NO_COMPRESSION:
		default:
			//Save raw data
			file.write(uncompressed, uncompressedSize);
			break;
		}

		file.close();
	}

	return ratio;
}

void saveRGBDFrameImagesToFiles(string filename, RGBDFramePtr frame)
{
	saveRGBDFrameImagesToFiles(filename, frame, NO_COMPRESSION, NO_COMPRESSION);
}

void saveRGBDFrameImagesToFiles(string filename, RGBDFramePtr frame, COMPRESSION_METHOD rgbCompression, COMPRESSION_METHOD depthCompression)
{
	if(frame->hasColor())
	{
		//Write color file
		char* rgbData = (char*)frame->getColorArray().get();
		int memSize = frame->getXRes()*frame->getYRes()*sizeof(ColorPixel);
		saveToCompressedBinaryFile(filename + ".rgb", rgbData, memSize, rgbCompression);

	}

	if(frame->hasDepth())
	{
		//write depth file
		char* depthData = (char*)frame->getDepthArray().get();
		int memSize = frame->getXRes()*frame->getYRes()*sizeof(DPixel);
		saveToCompressedBinaryFile(filename + ".depth", depthData, memSize, depthCompression);
	}
}


void loadRGBDFrameImagesFromFiles(string filename, RGBDFramePtr frame)
{
	loadColorImageFromFile(filename,frame);
	loadDepthImageFromFile(filename,frame);
}


void loadColorImageFromFile(string filename, RGBDFramePtr frame)
{
	ifstream rgbfile (filename+".rgb", ios::in|ios::binary);
	if (rgbfile.is_open())
	{
		char* rgbData = (char*)frame->getColorArray().get();
		int memSize = frame->getXRes()*frame->getYRes()*sizeof(ColorPixel);
		rgbfile.read(rgbData, memSize);
		rgbfile.close();
		frame->setHasColor(true);
	}
}

void loadDepthImageFromFile(string filename, RGBDFramePtr frame)
{
	ifstream depthfile (filename+".depth", ios::in|ios::binary);
	if (depthfile.is_open())
	{
		char* depthData = (char*)frame->getDepthArray().get();
		int memSize = frame->getXRes()*frame->getYRes()*sizeof(DPixel);
		depthfile.read(depthData, memSize);
		depthfile.close();
		frame->setHasDepth(true);
	}
}


bool makeDir(string directory)
{
	boost::filesystem::path dir(directory.c_str());
	return boost::filesystem::create_directories(dir);
}

bool isDirectoryEmpty(string directory)
{
	boost::filesystem::path dir(directory.c_str());
	if(boost::filesystem::is_directory(dir))
	{
		return boost::filesystem::is_empty(dir);
	}

	return false;
}


bool isDirectory(string dir)
{
	return boost::filesystem::is_directory(dir);
}


bool fileExists(string filename)
{
	boost::filesystem::path file(filename.c_str());
	return boost::filesystem::exists(file);
}


string loadTextFile(string filename)
{
	ifstream logfile (filename);

	if (logfile.is_open())
	{
		std::string str;

		logfile.seekg(0, std::ios::end);   
		str.reserve(logfile.tellg());
		logfile.seekg(0, std::ios::beg);

		str.assign((std::istreambuf_iterator<char>(logfile)),
			std::istreambuf_iterator<char>());
		return str;
	}
	return "";
}