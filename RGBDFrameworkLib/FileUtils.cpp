#include "FileUtils.h"

namespace rgbd
{
	namespace framework
	{

		string getCompressionMethodTag(COMPRESSION_METHOD method)
		{
			switch(method)
			{
			case LZ4_COMPRESSION:
				return string("lz4");
			case NO_COMPRESSION:
			default:
				return string("");
			}
		}

		COMPRESSION_METHOD getCompressionMethodFromTag(string tag)
		{
			if(tag.compare("lz4") == 0)
				return LZ4_COMPRESSION;

			return NO_COMPRESSION;
		}

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
				case LZ4_COMPRESSION:
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
					ratio = 1.0;
					break;
				}

				file.close();
			}

			return ratio;
		}


		void loadCompressedBinaryFile(string filename, char* outputArray, int memSize, COMPRESSION_METHOD compressMode)
		{
			char* compressed;
			ifstream file (filename, ios::in|ios::binary);
			if (file.is_open())
			{

				// get length of file:
				file.seekg (0, file.end);
				int length = (int) file.tellg();
				file.seekg (0, file.beg);

				switch(compressMode)
				{
				case LZ4_COMPRESSION:
					//Allocate memory for compressed data
					compressed = new char[length];
					file.read(compressed, length);
					LZ4_decompress_safe(compressed, outputArray, length, memSize);
					delete compressed;
					break;
				case NO_COMPRESSION:
				default:
					//read raw data
					file.read(outputArray, min(memSize, length));
					break;
				}
				file.close();
			}
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
			loadRGBDFrameImagesFromFiles(filename, frame, NO_COMPRESSION, NO_COMPRESSION);
		}


		void loadRGBDFrameImagesFromFiles(string filename, RGBDFramePtr frame, COMPRESSION_METHOD rgbCompression, COMPRESSION_METHOD depthCompression)
		{
			loadColorImageFromFile(filename,frame, rgbCompression);
			loadDepthImageFromFile(filename,frame, depthCompression);
		}

		void loadColorImageFromFile(string filename, RGBDFramePtr frame)
		{
			loadColorImageFromFile(filename,frame,NO_COMPRESSION);
		}

		void loadColorImageFromFile(string filename, RGBDFramePtr frame, COMPRESSION_METHOD compressionMode)
		{
			char* rgbData = (char*)frame->getColorArray().get();
			int memSize = frame->getXRes()*frame->getYRes()*sizeof(ColorPixel);
			loadCompressedBinaryFile(filename+".rgb", rgbData, memSize, compressionMode);
			frame->setHasColor(true);
		}


		void loadDepthImageFromFile(string filename, RGBDFramePtr frame)
		{
			loadDepthImageFromFile(filename, frame, NO_COMPRESSION);
		}

		void loadDepthImageFromFile(string filename, RGBDFramePtr frame, COMPRESSION_METHOD compressionMode)
		{
			char* depthData = (char*)frame->getDepthArray().get();
			int memSize = frame->getXRes()*frame->getYRes()*sizeof(DPixel);
			loadCompressedBinaryFile(filename+".depth", depthData, memSize, compressionMode);
			frame->setHasDepth(true);
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
	}
}