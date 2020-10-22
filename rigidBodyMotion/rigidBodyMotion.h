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

Eigen::Matrix<float, 2, 2> rigidBodyMotion(VectorOfDescriptorVector &descriptor1,
                                           VectorOfDescriptorVector &descriptor2,
                                           Eigen::Matrix<float, 3, Eigen::Dynamic> &matchProposal,
                                           Eigen::Matrix<float, 1, Eigen::Dynamic> optimizedAssociationSolution);

Eigen::Matrix<float, 2, 2> computeCrossCorrelationMatrix(Vector2fVector &xPrime,
                                                         Vector2fVector &yPrime,
                                                         Eigen::Matrix<float, 3, Eigen::Dynamic> &matchProposal,
                                                         Eigen::Matrix<float, 1, Eigen::Dynamic> optimizedAssociationSolution);

Eigen::Vector2f meanScan(VectorOfDescriptorVector &descriptor);
Vector2fVector positionPrime(VectorOfDescriptorVector &descriptor, Eigen::Vector2f meanScan);
