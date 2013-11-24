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


bool makeDir(string dir);

bool isDirectoryEmpty(string dir);