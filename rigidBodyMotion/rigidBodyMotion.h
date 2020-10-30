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

Eigen::Matrix<float, 2, 2> rigidBodyMotionSurf(vector<KeyPoint> keypoints1,
                                               vector<KeyPoint> keypoints2,
                                               vector< DMatch > matches);

Eigen::Matrix<float, 2, 2> rigidBodyMotion(VectorOfDescriptorVector &descriptor1,
                                           VectorOfDescriptorVector &descriptor2,
                                           Eigen::Matrix<float, 3, Eigen::Dynamic> &matchProposal,
                                           Eigen::Matrix<float, 1, Eigen::Dynamic> optimizedAssociationSolution);

Eigen::Matrix<float, 2, 2> computeCrossCorrelationMatrixSurf(Vector2fVector &xPrime,
                                                             Vector2fVector &yPrime);

Eigen::Matrix<float, 2, 2> computeCrossCorrelationMatrix(Vector2fVector &xPrime,
                                                         Vector2fVector &yPrime,
                                                         Eigen::Matrix<float, 3, Eigen::Dynamic> &matchProposal,
                                                         Eigen::Matrix<float, 1, Eigen::Dynamic> optimizedAssociationSolution);

Eigen::Vector2f meanScan(VectorOfDescriptorVector &descriptor);
Eigen::Vector2f meanScanSurf1(vector<KeyPoint> keypoints, vector< DMatch > matches);
Eigen::Vector2f meanScanSurf2(vector<KeyPoint> keypoints, vector< DMatch > matches);

Eigen::Vector4f meanScanWithAssociation(VectorOfDescriptorVector &descriptor1,
                                        VectorOfDescriptorVector &descriptor2,
                                        Eigen::Matrix<float, 3, Eigen::Dynamic> &matchProposal,
                                        Eigen::Matrix<float, 1, Eigen::Dynamic> &associationSolution);

Vector2fVector positionPrimeSurf1(vector<KeyPoint> keypoints, vector< DMatch > matches, Eigen::Vector2f meanScan);
Vector2fVector positionPrimeSurf2(vector<KeyPoint> keypoints, vector< DMatch > matches, Eigen::Vector2f meanScan);
Vector2fVector positionPrime(VectorOfDescriptorVector &descriptor, Eigen::Vector2f meanScan);
