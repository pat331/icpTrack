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
using namespace pr;

Mat keyPointExtraction(Mat radarScanImage, int maxNumberKeyPoint);

Mat prewittOperator(Mat radarScanImage);
Mat getMatrixSPrime(Mat radarScanImage);
Mat getMatrixH(Mat prewittImage, Mat SPrime);
Vector3fVector getIndicesOfElementsInDescendingOrder(Mat prewittImage);
Eigen::Vector2f findRangeBoundaries(float a, float r, Mat SPrime);
