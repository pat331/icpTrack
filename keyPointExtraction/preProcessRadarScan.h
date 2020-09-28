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

Mat prewittOperator(Mat radarScanImage);
Mat getMatrixSPrime(Mat radarScanImage);
// Mat getMatrixH(Mat prewittImage, Mat SPrime);
void getIndicesOfElementsInDescendingOrder(Mat prewittImage);
