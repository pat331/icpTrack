#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>

#include "dataAssociationRadar.h"
#include "defs.h"


using namespace std;
using namespace cv;
using namespace pr;

void unaryMatchesFromDescriptors(Mat L1, Mat L2){
  Vector2fVector unaryMatches;
}
/*VectorOfDescriptorVector*/
void createDescriptor(Mat L){

  Eigen::Vector2f positionLandMark;
  Vector2fVector landmarksPositionPolar;
  Vector2fVector landmarksPositionCart;
  Vector2fVector landmarksPositionPolarInFrameLandmark;
  int positionHelper;

  std::vector<float> angularHistogram(400,0);
  std::vector<float> annulusHistogram(7000,0);

  landmarksPositionPolar = getLandMarkPolarCoord(L);
  std::cerr << "landmarksPositionPolar"<< landmarksPositionPolar.size() << '\n';
  for (int i = 0; i < landmarksPositionPolar.size(); i++) {
    landmarksPositionPolarInFrameLandmark = getLandMarkPolarCoordInLandMarkFrame(landmarksPositionPolar, i);
    std::fill(angularHistogram.begin(), angularHistogram.end(), 0);
    std::fill(annulusHistogram.begin(), annulusHistogram.end(), 0);

    for (int k = 0; k < landmarksPositionPolarInFrameLandmark.size()-1; k++) {

      angularHistogram[(int)landmarksPositionPolarInFrameLandmark[k](1)]=+1;
      annulusHistogram[(int)landmarksPositionPolarInFrameLandmark[k](0)]=+1;
    }

  }

}

Vector2fVector getLandMarkPolarCoord(Mat L){
  std::cerr << "L size " << L.size() << '\n';
  Eigen::Vector2f positionLandMark;
  Vector2fVector positionPolar;

  for (int x = 0; x < L.cols; x++) {
    for (int y = 0; y < L.rows; y++) {
      if (L.at<float>(Point(x, y)) > 0) {  // Se il punto e' un landmark procediamo a calcolare il desciptor
        positionLandMark(0) = x;
        positionLandMark(1) = y;
        positionPolar.push_back(positionLandMark); // x = rho, y = theta
      }
    }
  }
  return positionPolar;
}

Vector2fVector getLandMarkPolarCoordInLandMarkFrame(Vector2fVector positionsLandMark, int landmark){
  Vector2fVector polarPositionsInFrameLand;
  Eigen::Vector2f positionLandMarkCartInFrame;
  Eigen::Vector2f positionLandMarkPolInFrame;
  Eigen::Vector2f newFrameCenter;

  float theta, rho;
  rho = positionsLandMark[landmark](0);
  theta = positionsLandMark[landmark](1)*angleResolution;
  newFrameCenter(0)= rho*cos(theta);
  newFrameCenter(1)= rho*sin(theta);

  for (int j = 0; j < positionsLandMark.size(); j++) {

    if (j==landmark) {
      continue;
    }
    rho = positionsLandMark[j](0);
    theta = positionsLandMark[j](1)*angleResolution;

    positionLandMarkCartInFrame(0) = rho*cos(theta) + newFrameCenter(0);
    positionLandMarkCartInFrame(1) = rho*sin(theta) + newFrameCenter(1);

    positionLandMarkPolInFrame(0) = sqrt(pow(positionLandMarkCartInFrame(0),2) + pow(positionLandMarkCartInFrame(1),2));
    positionLandMarkPolInFrame(1) = (float) atan2((double)positionLandMarkCartInFrame(1), (double)positionLandMarkCartInFrame(0));
    if (positionLandMarkPolInFrame(1)<0) {
      positionLandMarkPolInFrame(1) += 2*3.14;
    }
    positionLandMarkPolInFrame(1) = positionLandMarkPolInFrame(1)/angleResolution;
    positionLandMarkPolInFrame(1) = floor(positionLandMarkPolInFrame(1));
    polarPositionsInFrameLand.push_back(positionLandMarkPolInFrame);
  }

  return polarPositionsInFrameLand;
}
