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
using namespace pr;

LocalMap::LocalMap(){
  _originImage(0) = 0;
  _originImage(1) = 0;
  numeroDescrittori = 0;
  numeroScan = 0;
}

void LocalMap::initFirstMap(const std::vector<KeyPoint>& keypoints, const Mat& descriptors){
  _mapPoint = keypoints; // Hold the map
  _mapPointDescriptors= descriptors;
  numeroDescrittori += keypoints.size();
  numeroScan = 1;
  // Bring the origin to (0,0)
  for (size_t i = 0; i < _mapPoint.size(); i++) {
    _mapPoint[i].pt.x = _mapPoint[i].pt.x - _originImage(0);
    _mapPoint[i].pt.y = _mapPoint[i].pt.y - _originImage(1);
  }
  Eigen::Vector2f initRobotPose;
  initRobotPose << 400,400;
  _robotPose.push_back(initRobotPose);


}

void LocalMap::dispMap(){

  RGBImage local_image(1500, 1500);
  local_image.create(1500, 1500);
  local_image=cv::Vec3b(255,255,255);
  Vector2fVector provaDisp;
  std::vector<KeyPoint> keyForDisp = _mapPoint;
  for (size_t i = 0; i < _mapPoint.size(); i++) {
    Eigen::Vector2f disp;
    disp(0) = _mapPoint[i].pt.x*1;
    disp(1) = _mapPoint[i].pt.y*1;
    provaDisp.push_back(disp);
  }

  drawPoints(local_image, provaDisp, cv::Scalar(255,0,0),1);
  drawPoints(local_image, _robotPose, cv::Scalar(0,0,255),1);
  cv::imshow("Scan matcher", local_image);
  waitKey();
}


std::vector<DMatch> LocalMap::matchingWithMap(const std::vector<KeyPoint>& keypointsFrame,
                                              const Mat& descriptorsFrame){

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
  return ultimate_matches;

                   }

SE2 LocalMap::trackLocalMap(const std::vector<KeyPoint>& keypointsFrame1,
                             const Mat& descriptorsFrame1,
                             const std::vector<KeyPoint>& keypointsFrame,
                             const Mat& descriptorsFrame,
                             const SE2 scanMotion,
                             const std::vector<DMatch>& matchWithMap,
                             const std::vector<int>& associatedLandmarkIndex){

  // std::vector<KeyPoint> clippedKeypoints;
  // Mat clippedDescriptors;
   numeroScan += 1;
   std::cerr << "numero scan "<< numeroScan << '\n';
  //  numeroDescrittori += keypointsFrame1.size();
  //  int toClip = 400 * (numeroScan-1);
  //  if (numeroScan > 2) {
  //
  //    for (int i = toClip; i < _mapPoint.size(); i++) {
  //      clippedKeypoints.push_back(_mapPoint[i]);
  //      Mat newRow = _mapPointDescriptors.row(i);
  //      clippedDescriptors.push_back(newRow);
  //    }
  //  }else{
  //   clippedKeypoints = keypointsFrame1;
  //    clippedDescriptors = descriptorsFrame1;
  //
  //  }

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptorsFrame1, descriptorsFrame, knn_matches, 2 );
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


    std::vector<int> maxClique;
    maxClique = createConsistencyMatrix(keypointsFrame1, keypointsFrame, good_matches);
    // maxClique = createConsistencyMatrix(keypoints_object, keypoints_scene, good_matchesORB);

    std::vector<DMatch> ultimate_matches;
    for (size_t i = 0; i < good_matches.size(); i++) {
      if (maxClique[i] == 1) {
        ultimate_matches.push_back(good_matches[i]);
      }
    }



  Ptr<DescriptorMatcher> matcher2 = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  std::vector< std::vector<DMatch> > knn_matches2;
  matcher2->knnMatch( _mapPointDescriptors, descriptorsFrame, knn_matches2, 2 );
  //-- Filter matches using the Lowe's ratio test
  // const float ratio_thresh = 0.7f;
  std::vector<DMatch> good_matches2;
  for (size_t i = 0; i < knn_matches2.size(); i++)
  {
      if (knn_matches2[i][0].distance < ratio_thresh * knn_matches2[i][1].distance)
      {
          good_matches2.push_back(knn_matches2[i][0]);
      }
  }

  std::vector<KeyPoint> prova = keypointsFrame;
  Eigen::Vector2f trasformed;
  Eigen::Vector2f frameCoord;


  for (size_t i = 0; i < keypointsFrame.size(); i++) {
    // if (associatedLandmarkIndex[i] == 0) {
      frameCoord << keypointsFrame[i].pt.x, keypointsFrame[i].pt.y;
      trasformed = scanMotion.R*frameCoord + scanMotion.t;
      prova[i].pt.x = trasformed(0);
      prova[i].pt.y = trasformed(1);

    // }
  }

  std::vector<int> maxClique2;
  maxClique2 = createConsistencyMatrix(_mapPoint, prova, good_matches2);
  // maxClique = createConsistencyMatrix(keypoints_object, keypoints_scene, good_matchesORB);
  // const float ratio_thresh = 0.7f;
  std::vector<DMatch> ultimate_matches2;
  for (size_t i = 0; i < good_matches2.size(); i++) {
    if (maxClique2[i] == 1) {
      ultimate_matches2.push_back(good_matches2[i]);
    }
  }


  std::cerr << "good_matches "<< good_matches.size() << '\n';
  std::cerr << "### good_matches "<< good_matches2.size() << '\n';
  std::cerr << "ultimate_matches size trackLocalMap "<< ultimate_matches.size() << '\n';
  std::cerr << "###ultimate_matches size trackLocalMap "<< ultimate_matches2.size() << '\n';
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
  // Ptr<DescriptorMatcher> matcherFrameMap = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  // std::vector< std::vector<DMatch> > knn_matches;
  // matcherFrameMap->knnMatch(_mapPointDescriptors, descriptorsFrame, knn_matches, 2 );
  // //-- Filter matches using the Lowe's ratio test
  // const float ratio_thresh = 0.7f;
  // std::vector<DMatch> good_matches;
  // for (size_t i = 0; i < knn_matches.size(); i++)
  // {
  //     if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
  //     {
  //         good_matches.push_back(knn_matches[i][0]);
  //     }
  // }
  // std::cerr << "size of good mathces in localmap "<< good_matches.size() << '\n';
  //
  // std::vector<int> maxClique;
  // maxClique = createConsistencyMatrix( _mapPoint, keypointsFrame, good_matches);
  // std::vector<DMatch> ultimate_matches;
  // for (size_t i = 0; i < good_matches.size(); i++) {
  //   if (maxClique[i] == 1) {
  //     ultimate_matches.push_back(good_matches[i]);
  //   }
  // }
  //
  // std::cerr << "ultimate matches size in local track "<<ultimate_matches.size() << '\n';
  // if (ultimate_matches.size() >= 3) {
  //   mergeMap(keypointsFrame, ultimate_matches, RotationMatrix, translationVector);
  // }
  SE2 finalMotion;
  Eigen::Matrix<float, 2, 2> Rf;
  Eigen::Matrix<float, 2, 2> Rf2;
  Eigen::Matrix<float, 2, 2> R2;

  Rf = rigidBodyMotionSurf(keypointsFrame1, keypointsFrame, ultimate_matches); // questa è la rotazione che porta da ref1->ref2
  Rf2 = rigidBodyMotionSurf(_mapPoint, prova, ultimate_matches2);
  // std::cerr << "Rf "<< Rf << '\n';
  // std::cerr << "### Rf ### "<< Rf2 << '\n';
  Eigen::Vector2f mean1;
  mean1 = meanScanSurf1(keypointsFrame1, ultimate_matches);
  // std::cerr << "mean1  "<< mean1 << '\n';

  Eigen::Vector2f mean1b;
  mean1b = meanScanSurf1(_mapPoint, ultimate_matches2);
  // std::cerr << "### mean1 ###  "<< mean1b << '\n';

  Eigen::Vector2f mean2;
  mean2 = meanScanSurf2(keypointsFrame, ultimate_matches);
  // std::cerr << "mean2  "<< mean2 << '\n';

  Eigen::Vector2f mean2b;
  mean2b = meanScanSurf2(prova, ultimate_matches2);
  // std::cerr << "### mean2 ### "<< mean2b << '\n';


  Eigen::Vector2f translationVectorf;
  Eigen::Vector2f t2;
  translationVectorf = mean2b - Rf2 * mean1b;
  // std::cerr << "translationVectorf "<<translationVectorf << '\n';
  // translationVectorTot += translationVectorf;

  R2 = Rf.inverse();
  t2 = -R2*translationVectorf;
  finalMotion.R = scanMotion.R*R2;
  finalMotion.t = scanMotion.R*t2+scanMotion.t;
  // std::cerr << "/* finalMotion.t */"<< finalMotion.t << '\n';


  std::vector<KeyPoint> keyFrameTrasformed = keypointsFrame;
  Eigen::Vector2f trasformedPoint;
  Eigen::Vector2f pointInFrameCoord;
  float meanPosX;
  float meanPosY;

  for (size_t i = 0; i < keypointsFrame.size(); i++) {
    pointInFrameCoord << keypointsFrame[i].pt.x, keypointsFrame[i].pt.y;
    trasformedPoint = finalMotion.R*pointInFrameCoord + finalMotion.t;
    keyFrameTrasformed[i].pt.x = trasformedPoint(0);
    keyFrameTrasformed[i].pt.y = trasformedPoint(1);
    if (associatedLandmarkIndex[i] == 0) {
      _mapPoint.push_back(keyFrameTrasformed[i]);
      Mat newRow = descriptorsFrame.row(i);
      _mapPointDescriptors.push_back(newRow);
    }
  }
  std::cerr << "matchWithMap size "<< matchWithMap.size() << '\n';
  for (size_t i = 0; i < ultimate_matches2.size(); i++) {

    meanPosX = (keyFrameTrasformed[ultimate_matches2[i].trainIdx].pt.x + _mapPoint[ultimate_matches2[i].queryIdx].pt.x)/2;
    meanPosY = (keyFrameTrasformed[ultimate_matches2[i].trainIdx].pt.y + _mapPoint[ultimate_matches2[i].queryIdx].pt.y)/2;
    _mapPoint[ultimate_matches2[i].queryIdx].pt.x = meanPosX;
    _mapPoint[ultimate_matches2[i].queryIdx].pt.y = meanPosY;
    // std::cerr << "entro dentro merge map " << '\n';
    // std::cerr << "diff on x " << keyFrameTrasformed[matchWithMap[i].trainIdx].pt.x - _mapPoint[matchWithMap[i].queryIdx].pt.x << '\n';
    // std::cerr << "diff on y " << keyFrameTrasformed[matchWithMap[i].trainIdx].pt.y - _mapPoint[matchWithMap[i].queryIdx].pt.y << '\n';

  }
  return finalMotion;


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
                    const Mat& descriptorsFrame,
                    const Eigen::Matrix<float, 2, 2>& R,
                    const Eigen::Vector2f& t){
    // std::cerr << "number of map point"<< _mapPoint.size() << '\n';
    Eigen::Vector2f pointInFrameCoord;
    Eigen::Vector2f trasformedPoint;
    std::vector<KeyPoint> trasformedKeyPoint = keyFrame;
    for (size_t i = 0; i < keyFrame.size(); i++) {
      pointInFrameCoord << keyFrame[i].pt.x, keyFrame[i].pt.y;
      trasformedPoint = R*pointInFrameCoord + t;
      trasformedKeyPoint[i].pt.x = trasformedPoint(0);
      trasformedKeyPoint[i].pt.y = trasformedPoint(1);
      _mapPoint.push_back(trasformedKeyPoint[i]);
      Mat newRow = descriptorsFrame.row(i);
      _mapPointDescriptors.push_back(newRow);
    }
                    }

void LocalMap::robotMotion(const SE2& robMotion){
  // Retrieve the last robotPose
  Eigen::Vector2f lastRobotPose = _robotPose.back();
  Eigen::Vector2f newRobotPose = robMotion.R * lastRobotPose + robMotion.t;
  _robotPose.push_back(newRobotPose);

}
