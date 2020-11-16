#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>


#include "localMap.h"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

LocalMap::LocalMap(){
  _originImage(0) = 0;
  _originImage(1) = 0;
}

void LocalMap::initFirstMap(const std::vector<KeyPoint>& keypoints, const Mat& descriptors){
  _mapPoint = keypoints; // Hold the map
  _mapPointDescriptors= descriptors;
  // Bring the origin to (0,0)
  for (size_t i = 0; i < _mapPoint.size(); i++) {
    _mapPoint[i].pt.x = _mapPoint[i].pt.x - _originImage(0);
    _mapPoint[i].pt.y = _mapPoint[i].pt.y - _originImage(1);
  }

}

void LocalMap::dispMap(){

  RGBImage local_image(1500, 1500);
  local_image.create(1500, 1500);
  local_image=cv::Vec3b(255,255,255);
  std::vector<KeyPoint> keyForDisp = _mapPoint;
  for (size_t i = 0; i < _mapPoint.size(); i++) {
    keyForDisp[i].pt.x = _mapPoint[i].pt.x*1.4;
    keyForDisp[i].pt.y = _mapPoint[i].pt.y*1.4;
  }
  Mat mapPoints;
  drawKeypoints( local_image, keyForDisp, mapPoints );
  //-- Show detected (drawn) keypoints
  imshow("First map", mapPoints );
  waitKey();
}

void LocalMap::trackLocalMap(const std::vector<KeyPoint>& keypointsFrame,
                             const Mat& descriptorsFrame,
                             // const std::vector<DMatch>& matchesInFrame,
                             const Eigen::Matrix<float, 2, 2>& RotationMatrix,
                            const Eigen::Vector2f& translationVector){

  // Retrieve the non associated landmark
  // std::vector<int> notAssociatedLandmarkIndex(keypointsFrame.size(),0);
  // for (size_t i = 0; i < matchesInFrame.size(); i++) {
  //   notAssociatedLandmarkIndex[matchesInFrame[i].queryIdx] = 1;
  //   // all the zeros are the non associated ones
  // }
  // std::vector<KeyPoint> keyFrame;
  // for (size_t i = 0; i < keypointsFrame.size(); i++) {
  //   if (notAssociatedLandmarkIndex[i] == 0) {
  //     keyFrame.push_back(keypointsFrame[i]);
  //   }
  // }
  //-- Perform an association between frame and map
  //-- Matching descriptor vectors with a FLANN based matcher
  Ptr<DescriptorMatcher> matcherFrameMap = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  std::vector< std::vector<DMatch> > knn_matches;
  matcherFrameMap->knnMatch(_mapPointDescriptors, descriptorsFrame, knn_matches, 2 );
  //-- Filter matches using the Lowe's ratio test
  const float ratio_thresh = 0.7f;
  std::vector<DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++)
  {
      if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
      {
          good_matches.push_back(knn_matches[i][0]);
      }
  }
  // std::cerr << "size of good mathces in localmap "<< good_matches.size() << '\n';

  std::vector<int> maxClique;
  maxClique = createConsistencyMatrix( _mapPoint, keypointsFrame, good_matches);
  std::vector<DMatch> ultimate_matches;
  for (size_t i = 0; i < good_matches.size(); i++) {
    if (maxClique[i] == 1) {
      ultimate_matches.push_back(good_matches[i]);
    }
  }

  std::cerr << "ultimate matches size in local track "<<ultimate_matches.size() << '\n';
  if (ultimate_matches.size() >= 3) {
    mergeMap(keypointsFrame, ultimate_matches, RotationMatrix, translationVector);
  }



                   }

void LocalMap::mergeMap(const std::vector<KeyPoint>& keyFrame,
                        const std::vector<DMatch>& matchesForMerge,
                        const Eigen::Matrix<float, 2, 2>& RotationMatrix,
                        const Eigen::Vector2f& translationVector){

  std::vector<KeyPoint> keyFrameTrasformed = keyFrame;
  Eigen::Vector2f rotatedPoint;
  Eigen::Vector2f rotatedTranslation;
  rotatedTranslation << translationVector(0), translationVector(1);
  for (size_t i = 0; i < keyFrame.size(); i++) {
    rotatedPoint << keyFrame[i].pt.x, keyFrame[i].pt.y;
    rotatedTranslation << rotatedTranslation;
    rotatedPoint = RotationMatrix.inverse()*rotatedPoint;
    keyFrameTrasformed[i].pt.x = rotatedPoint(0) - rotatedTranslation(0);
    keyFrameTrasformed[i].pt.y = rotatedPoint(1) - rotatedTranslation(1);
  }
  float meanPosX;
  float meanPosY;
  for (size_t i = 0; i < matchesForMerge.size(); i++) {
    meanPosX = (keyFrameTrasformed[matchesForMerge[i].queryIdx].pt.x + _mapPoint[matchesForMerge[i].trainIdx].pt.x)/2;
    meanPosY = (keyFrameTrasformed[matchesForMerge[i].queryIdx].pt.y + _mapPoint[matchesForMerge[i].trainIdx].pt.y)/2;
    _mapPoint[matchesForMerge[i].trainIdx].pt.x = meanPosX;
    _mapPoint[matchesForMerge[i].trainIdx].pt.y = meanPosY;
    std::cerr << "entro dentro merge map " << '\n';
    std::cerr << "diff on x " << keyFrameTrasformed[matchesForMerge[i].trainIdx].pt.x - _mapPoint[matchesForMerge[i].queryIdx].pt.x << '\n';
    std::cerr << "diff on y " << keyFrameTrasformed[matchesForMerge[i].trainIdx].pt.y - _mapPoint[matchesForMerge[i].queryIdx].pt.y << '\n';

  }

                          }

void LocalMap::insertKeyFrame(const std::vector<KeyPoint>& keyFrame,
                    const Eigen::Matrix<float, 2, 2>& R,
                    const Eigen::Vector2f& t){

    Eigen::Vector2f pointInFrameCoord;
    Eigen::Vector2f trasformedPoint;
    std::vector<KeyPoint> trasformedKeyPoint = keyFrame;
    for (size_t i = 0; i < keyFrame.size(); i++) {
      pointInFrameCoord << keyFrame[i].pt.x, keyFrame[i].pt.y;
      trasformedPoint = R*pointInFrameCoord + t;
      trasformedKeyPoint[i].pt.x = trasformedPoint(0);
      trasformedKeyPoint[i].pt.y = trasformedPoint(1);
      _mapPoint.push_back(trasformedKeyPoint[i]);
    }
                    }
