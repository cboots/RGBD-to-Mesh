#pragma once


#include <stdio.h>
#include <ostream>
#include <fstream>
#include <istream>
#include <string>
#include "RGBDFrame.h"
#include <boost/filesystem.hpp>
#include "lz4/lz4.h"
#include <iostream>

using namespace std;


namespace rgbd
{
	namespace framework
	{
		//Enumeration of supported binary compression methods
		enum COMPRESSION_METHOD {NO_COMPRESSION = 0, LZ4_COMPRESSION = 1};

		string getCompressionMethodTag(COMPRESSION_METHOD method);
		COMPRESSION_METHOD getCompressionMethodFromTag(string tag);

		//Saves image files with appropriate file extensions and no compression
		void saveRGBDFrameImagesToFiles(string filename, RGBDFramePtr frame);

		//Saves image files with appropriate file extensions and with the specified compression schemes.
		//Depth and Color data may be compressed with different methods
		void saveRGBDFrameImagesToFiles(string filename, RGBDFramePtr frame, COMPRESSION_METHOD rgbCompression, COMPRESSION_METHOD depthCompression);

		//Loads both .depth and .rgb files.
		//Must call frame.setResolution(x,y) before using this function.
		//Assumes the provided memory is large enough to store the entire frame
		//The filename should not include the file extension. (i.e. "1" not "1.rgb")
		//Filename can be relative or absolute path
		void loadRGBDFrameImagesFromFiles(string filename, RGBDFramePtr frame);

		//See void loadRGBDFrameImagesFromFiles(string filename, RGBDFramePtr frame) for usage
		void loadRGBDFrameImagesFromFiles(string filename, RGBDFramePtr frame, COMPRESSION_METHOD rgbCompression, COMPRESSION_METHOD depthCompression);

		//Loads just the color image from file. 
		//Must call frame.setResolution(x,y) before using this function.
		//Assumes the provided memory is large enough to store the entire frame
		//The filename should not include the file extension. (i.e. "1" not "1.rgb")
		//Filename can be relative or absolute path
		void loadColorImageFromFile(string filename, RGBDFramePtr frame);

		//See void loadColorImageFromFile(string filename, RGBDFramePtr frame);
		void loadColorImageFromFile(string filename, RGBDFramePtr frame, COMPRESSION_METHOD compressionMode);


		//Loads just the depth image from file. 
		//Must call frame.setResolution(x,y) before using this function.
		//Assumes the provided memory is large enough to store the entire frame
		//The filename should not include the file extension. (i.e. "1" not "1.rgb")
		//Filename can be relative or absolute path
		//Assumes no compression was used in saving the file
		void loadDepthImageFromFile(string filename, RGBDFramePtr frame);

		//See void loadDepthImageFromFile(string filename, RGBDFramePtr frame);
		void loadDepthImageFromFile(string filename, RGBDFramePtr frame, COMPRESSION_METHOD compressionMode);


		//Convenience function wrapper to make a directory on the local filesystem.
		bool makeDir(string dir);

		//Returns true if the provided path is a directory and the directory contains no files
		bool isDirectoryEmpty(string dir);

		//Returns true if the provided path is a directory
		bool isDirectory(string dir);

		//Returns true if the provided file exists
		bool fileExists(string filename);

		//Loads an entire text file into a c++ string object.
		//Warning: use at your own risk on huge files
		string loadTextFile(string filename);

	}
}