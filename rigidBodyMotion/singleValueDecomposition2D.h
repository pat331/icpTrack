#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>

#include "defs.h"

#include "Array.h"
#include "Array.hpp"
#include "def.h"

#define angleResolution  0.0157079;
using namespace cv;
using namespace std;
using namespace pr;

void singularValueDecomposition2D(Eigen::Matrix<float, 2, 2> R);
Eigen::Matrix<float, 2, 2> computationOfU(float a, float b, float c, float d);
Eigen::Matrix<float, 2, 2> computationOfSigma(float a, float b, float c, float d);
Eigen::Matrix<float, 2, 2> computationOfV(float a, float b, float c, float d);
