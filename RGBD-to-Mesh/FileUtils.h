#pragma once


#include <stdio.h>
#include <ostream>
#include <fstream>
#include <istream>
#include <string>
#include "RGBDFrame.h"

using namespace std;
void saveRGBDFrameImagesToFiles(string filename, RGBDFramePtr frame);

//Must call frame.setResolution(x,y) before using this function.
void loadRGBDFrameImagesFromFiles(string filename, RGBDFramePtr frame);