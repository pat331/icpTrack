#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>

#include "defs.h"

using namespace cv;
using namespace std;

//Cropping the radar scan Image
Mat cropRadarScan(Mat radarScanImage, int pointX = 11, int pointY = 0, int length = 3500, int height = 400);
//Radar scan Filtering
Mat radarScanFilter(Mat radarScanImage, double thresh = 28, double maxValue = 255, double thresholdCanny1 = 28, double thresholdCanny2 = 150);
//  Converto coordinate pixel in coordinate world
Eigen::Vector2f pixelToWorldCoord(Point pixelCoord, float pixelRange);

// Vector2f heuristicLandmarks(Vector2f landMarkCartesian);
