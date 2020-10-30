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
#include "DynArray.h"
#include "DynArray.hpp"
#include "FFTReal.h"
#include "OscSinCos.h"
#include "OscSinCos.hpp"

#define angleResolution  0.0157079;
using namespace cv;
using namespace std;
using namespace pr;

std::vector<int> createConsistencyMatrix(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector< DMatch > matches);
std::vector<int> Grasp(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &consistencyMatrix);
