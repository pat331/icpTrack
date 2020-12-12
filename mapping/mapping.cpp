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


#include "mapping.h"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace pr;

Mapping::Mapping(){
  _totalMotion.R << 1,0,0,1;
  _totalMotion.t << 0,0;

  _total.R << 1,0,0,1;
  _total.t << 0,0;

  _totalErrorRotation = 0;
  _totalErrorTranslation = 0;

  Eigen::Vector2f initRobotPose;
  initRobotPose << 400,400;
  _robotPose.push_back(initRobotPose);

  _pose << 400 ,400;
}

void Mapping::initMap(const KeyAndDesc& kd, const SE2& mappingMotion){
  _mapPoint = kd.keypoints;
  _mapDescriptors = kd.descriptors;

  _motion.R = mappingMotion.R;
  _motion.t = mappingMotion.t;

             }

void Mapping::dispMotion(){

  RGBImage local_image(1500, 1500);
  local_image.create(1500, 1500);
  local_image=cv::Vec3b(255,255,255);

  for (size_t i = 0; i < _robotPose.size(); i++) {
    // _robotPose[i] <<  ((_robotPose[i](0)*1)+400)*0.2, ((_robotPose[i](1)*1)+400)*0.2;
    _robotPose[i] <<  _robotPose[i](0), _robotPose[i](1);
  }

  drawPoints(local_image, _robotPose, cv::Scalar(0,0,255),1);
  cv::imshow("MAPPING", local_image);
  waitKey();
}

KeyAndDesc Mapping::findScanKeyPoint(const Mat& scan){

  int minHessian = 20;
  Ptr<SURF> detector = SURF::create( minHessian);
  std::vector<KeyPoint> keypoints;
  Mat descriptors;
  detector->detectAndCompute( scan, noArray(), keypoints, descriptors);

  KeyAndDesc kd;
  kd.keypoints = keypoints;
  for (size_t i = 0; i < kd.keypoints.size(); i++) {
    kd.keypoints[i].pt.x *= 0.5;
    kd.keypoints[i].pt.y *= 0.5;
    // float prova1 = kd.keypoints[i].pt.x;
    // float prova2 = kd.keypoints[i].pt.y;
    // kd.keypoints[i].pt.x *= 0.044;
    // kd.keypoints[i].pt.y *= 0.044;
  }
  kd.descriptors = descriptors;
  return kd;
}

std::vector<DMatch> Mapping::matchMap(const KeyAndDesc& kd){

  // Devo portarmi i keypoint nel riferimento della mappa.
  Eigen::Vector2f pointInFrameCoord;
  Eigen::Vector2f trasformedPoint;
  std::vector<KeyPoint> kd2MapFrame = kd.keypoints;
  for (size_t i = 0; i < kd2MapFrame.size(); i++) {
    pointInFrameCoord << kd2MapFrame[i].pt.x, kd2MapFrame[i].pt.y;
    trasformedPoint = _motion.R*pointInFrameCoord + _motion.t;
    kd2MapFrame[i].pt.x = trasformedPoint(0);
    kd2MapFrame[i].pt.y = trasformedPoint(1);
  }
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  std::vector< std::vector<DMatch> > knn_matches;
  matcher->knnMatch( _mapDescriptors, kd.descriptors, knn_matches, 2 );
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
  std::cerr << "good_matches size "<< good_matches.size() << '\n';

  std::vector<int> maxClique;
  maxClique = createConsistencyMatrix( _mapPoint, kd2MapFrame, good_matches);
  std::vector<DMatch> ultimate_matches;
  for (size_t i = 0; i < good_matches.size(); i++) {
    if (maxClique[i] == 1) {
      ultimate_matches.push_back(good_matches[i]);
    }
  }
  std::cerr << "  +++++++++++++++++++  " << '\n';
  std::cerr << " ultimate_matches size "<< ultimate_matches.size() << '\n';
  if (ultimate_matches.size()<6) {
    std::cerr << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << '\n';
    std::cerr << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << '\n';
    std::cerr << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << '\n';
  }
  _tooMuchPoint = ultimate_matches.size();
  return ultimate_matches;
}


SE2 Mapping::scanMap(const KeyAndDesc& kd2,
                     const std::vector<DMatch>& match){

  // Devo portarmi i keypoint nel riferimento della mappa.
  Eigen::Vector2f pointInFrameCoord;
  Eigen::Vector2f trasformedPoint;
  std::vector<KeyPoint> kd2MapFrame = kd2.keypoints;
  for (size_t i = 0; i < kd2MapFrame.size(); i++) {
    pointInFrameCoord << kd2MapFrame[i].pt.x, kd2MapFrame[i].pt.y;
    trasformedPoint = _motion.R*pointInFrameCoord + _motion.t;
    kd2MapFrame[i].pt.x = trasformedPoint(0);
    kd2MapFrame[i].pt.y = trasformedPoint(1);
  }

  Eigen::Matrix<float, 2, 2> Rf, R2, Rtot;
  Eigen::Vector2f t2, translationVectorTot;

  Rf = rigidBodyMotionSurf(_mapPoint, kd2MapFrame, match); // questa è la rotazione che porta da ref1->ref2
  // Rf << Rf(0,0)*0.8,Rf(0,1)*0.8,Rf(1,0)*0.8,Rf(1,1)*0.8;
  std::cerr << "   ^^^    " << '\n';
  std::cerr << " ROTATION " << Rf <<'\n';
  Eigen::Vector2f mean1;
  mean1 = meanScanSurf1(_mapPoint, match);
  Eigen::Vector2f mean2;
  mean2 = meanScanSurf2(kd2MapFrame, match);

  Eigen::Vector2f translationVectorf;
  translationVectorf = mean2 - Rf * mean1;
  std::cerr << "+++ translationVectorf +++ "<< translationVectorf << '\n';
  _highTranslation = translationVectorf;

  // Dont go further if there is essentially no motion
  if (sqrt(pow(translationVectorf(0),2)+pow(translationVectorf(1),2))<0.2) {
    std::cerr << "No MoTiOn" << '\n';
    SE2 noMotion;
    noMotion.R.setIdentity();
    noMotion.t << 0,0;
    return noMotion;
  }
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(Rf, Eigen::ComputeThinU | Eigen::ComputeThinV);
  // Eigen::JacobiSVD<Eigen::Matrix2f> svd(Rf, Eigen::ComputeThinU | Eigen::ComputeThinV);
  float det_uv = (svd.matrixU() * svd.matrixV().transpose()).determinant();
  // Eigen::Vector3f singular_values(1.f, 1.f, det_uv);
  Eigen::Vector2f singular_values( 1.f, det_uv);
  Rf = svd.matrixU() * singular_values.asDiagonal() * svd.matrixV().transpose();


  //////////////////////////////////////////////////////////////////////////////
  // Prova ICP
  // SMICPSolver solver;
  // // Put the vector match into a vector of pair
  // IntPairVector correspondences;
  // correspondences = matchPair(match);
  // // std::cerr << "correspondences size "<< correspondences<< '\n';
  // // Put the vector of keypoints of map and keyframe into a Vector2fVector
  // Vector2fVector mapPoints;
  // Vector2fVector framePoints;
  // for (size_t i = 0; i < _mapPoint.size(); i++) {
  //   Eigen::Vector2f mapPoint;
  //   mapPoint << _mapPoint[i].pt.x, _mapPoint[i].pt.y;
  //   mapPoints.push_back(mapPoint);
  // }
  // for (size_t i = 0; i < kd2MapFrame.size(); i++) {
  //   Eigen::Vector2f framePoint;
  //   framePoint << kd2MapFrame[i].pt.x, kd2MapFrame[i].pt.y;
  //   framePoints.push_back(framePoint);
  // }
  //
  // Eigen::Isometry2f initialGuess;
  // initialGuess.setIdentity();
  // for (size_t i = 0; i < 2; i++) {
  //   for (size_t j = 0; j < 2; j++) {
  //     initialGuess(i,j) = Rf(i,j);
  //   }
  // }
  // for (size_t i = 0; i < 2; i++) {
  //   initialGuess(i,2) = translationVectorf(i);
  // }
  // std::cerr << "initialGuess "<< initialGuess.matrix() << '\n';
  // // initialGuess.setIdentity();
  // // Icp solver initialization
  // solver.init(initialGuess, mapPoints, framePoints);
  // for (int n_round=0; n_round < 5; n_round++){
  //   solver.oneRound(correspondences, false);
  // }
  // Eigen::Isometry2f refinedMov;
  // refinedMov = solver.transform();
  // std::cerr << "REFINED MOV "<< refinedMov.matrix() << '\n';
  //////////////////////////////////////////////////////////////////////////////

  //Quantities for Error estimation
  _scanMotion.R = Rf;
  _scanMotion.t = translationVectorf;
  //
  // Map motion inversa: porto la mappa nel frame dello scan (parziale, perché dopo pochi scan la elimino)
  // _total.R = _total.R*Rf;
  // _total.t = _total.R*translationVectorf+_total.t;

  R2 = Rf.inverse();
  t2 = -R2*translationVectorf;
  std::cerr << " °° t2 °° "<< t2 << '\n';

  Rtot = _motion.R*R2;
  translationVectorTot = _motion.R*t2+_motion.t;
  _motion.R = Rtot;
  _motion.t = translationVectorTot;

  // Total motion (motion del robot: totale la porto fino alla fine)
  Rtot = _totalMotion.R*R2;
  translationVectorTot = _totalMotion.R*t2+_totalMotion.t;
  _totalMotion.R = Rtot;
  _totalMotion.t = translationVectorTot;

  std::cerr << " §§§§§§§§§§ " << '\n';
  std::cerr << "total rotation "<< _totalMotion.R << '\n';
  std::cerr << "total motion "<< _totalMotion.t << '\n';


  SE2 robMotion;
  robMotion.R = R2;
  robMotion.t = t2;

  return robMotion;

            }

void Mapping::robotMotion(){

  Eigen::Vector2f initRob;
  initRob << 400,400;
  std::cerr << "prova robot pose "<< _totalMotion.R*initRob << '\n';
  Eigen::Vector2f newRobotPose = _totalMotion.R*initRob + _totalMotion.t;
  std::cerr << "  °°°°°°°°° " << '\n';
  std::cerr << " newRobotPose "<< newRobotPose << '\n';
  _robotPose.push_back(newRobotPose);

  // Eigen::Vector2f initRob;
  // initRob << 400,400;
  // std::cerr << "prova robot pose "<< _total.R*initRob << '\n';
  // Eigen::Vector2f newRobotPose = _total.R*initRob + _total.t;
  // std::cerr << "  °°°°°°°°° " << '\n';
  // std::cerr << " newRobotPose "<< newRobotPose << '\n';
  // _robotPose.push_back(newRobotPose);

}

void Mapping::insertKeyFrame(const KeyAndDesc& kd2){

  Eigen::Vector2f pointInFrameCoord;
  Eigen::Vector2f trasformedPoint;
  std::vector<KeyPoint> trasformedKeyPoint = kd2.keypoints;
  for (size_t i = 0; i < kd2.keypoints.size(); i++) {
    pointInFrameCoord << trasformedKeyPoint[i].pt.x, trasformedKeyPoint[i].pt.y;
    trasformedPoint = _motion.R*pointInFrameCoord + _motion.t;
    trasformedKeyPoint[i].pt.x = trasformedPoint(0);
    trasformedKeyPoint[i].pt.y = trasformedPoint(1);
    _mapPoint.push_back(trasformedKeyPoint[i]);
    Mat newRow = kd2.descriptors.row(i);
    _mapDescriptors.push_back(newRow);
  }

}

void Mapping::insertFrameReset(const KeyAndDesc& kd2, const SE2& motion){

  Eigen::Vector2f pointInFrameCoord;
  Eigen::Vector2f trasformedPoint;
  std::vector<KeyPoint> trasformedKeyPoint = kd2.keypoints;
  for (size_t i = 0; i < kd2.keypoints.size(); i++) {
    pointInFrameCoord << trasformedKeyPoint[i].pt.x, trasformedKeyPoint[i].pt.y;
    trasformedPoint = motion.R*pointInFrameCoord + motion.t;
    trasformedKeyPoint[i].pt.x = trasformedPoint(0);
    trasformedKeyPoint[i].pt.y = trasformedPoint(1);
    _mapPoint.push_back(trasformedKeyPoint[i]);
    Mat newRow = kd2.descriptors.row(i);
    _mapDescriptors.push_back(newRow);
  }

}

bool Mapping::resetMap(){

  if (sqrt(pow(_motion.t(0),2) + pow(_motion.t(1),2)) > 70*0.5 || _tooMuchPoint > 200) {
    std::cerr << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << '\n';
    std::cerr << " !!!!!!! RESET MAP !!!!!!! " << '\n';
    return 1;
  }else{
    return 0;
  }
}

void Mapping::clearMap(){
  _mapPoint.clear();
  _mapDescriptors = Mat();
  _motion.R << 1,0,0,1;
  _motion.t << 0,0;
}


void Mapping::errorEstimation(const Eigen::Isometry2f& isometry_gt,
                              const Eigen::Isometry2f& isometry_calc){


  Eigen::Matrix<float, 2, 2> deltaRk;
  deltaRk = isometry_gt.linear()*isometry_calc.linear().transpose();
  float deltaPhik;
  deltaPhik = atan2(deltaRk(1,0), deltaRk(0,0));

  Eigen::Vector2f translationConverted;
  translationConverted = deltaRk * isometry_calc.translation();
  // translationConverted *= 0.0438;

  float deltaPk;
  deltaPk = pow( isometry_gt.translation()(0) - translationConverted(0), 2) + pow( isometry_gt.translation()(1) - translationConverted(1), 2);

  _totalErrorTranslation += deltaPk;
  _totalErrorRotation += deltaPhik;

                     }

void Mapping::dispError(){
  std::cerr << "_totalErrorTranslation "<< _totalErrorTranslation << '\n';
  std::cerr << "_totalErrorRotation "<< _totalErrorRotation << '\n';
}
