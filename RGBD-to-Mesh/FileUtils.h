#pragma once


#include <stdio.h>
#include <ostream>
#include <fstream>
#include <istream>
#include <string>
#include "RGBDFrame.h"
#include <boost/filesystem.hpp>

using namespace std;
void saveRGBDFrameImagesToFiles(string filename, RGBDFramePtr frame);

//Must call frame.setResolution(x,y) before using this function.
void loadRGBDFrameImagesFromFiles(string filename, RGBDFramePtr frame);
void loadColorImageFromFile(string filename, RGBDFramePtr frame);
void loadDepthImageFromFile(string filename, RGBDFramePtr frame);


bool makeDir(string dir);

bool isDirectoryEmpty(string dir);


bool isDirectory(string dir);

bool fileExists(string filename);

string loadTextFile(string filename);
