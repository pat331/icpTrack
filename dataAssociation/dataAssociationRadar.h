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

// nota bene: questa funzione in realta' deve ritornare un Vector2fVector
void unaryMatchesFromDescriptors(Mat L1, Mat L2);

/*VectorOfDescriptorVector*/
void createDescriptor(Mat L);
Vector2fVector getLandMarkPolarCoord(Mat L);
Vector2fVector getLandMarkPolarCoordInLandMarkFrame(Vector2fVector positionLsandMark, int landmark);
