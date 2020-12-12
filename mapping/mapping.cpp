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
using namespace Eigen;
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
    // kd.keypoints[i].pt.x *= 0.5;
    // kd.keypoints[i].pt.y *= 0.5;
    // float prova1 = kd.keypoints[i].pt.x;
    // float prova2 = kd.keypoints[i].pt.y;
    kd.keypoints[i].pt.x *= 0.044;
    kd.keypoints[i].pt.y *= 0.044;
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
  
  // tg use holy eigen structure to represent 2d vectors
  using Vector2fVector = std::vector<Vector2f,Eigen::aligned_allocator<Vector2f>>;
  Vector2fVector map_points, scan_points;
  // tg extract number of points involved in the match
  // and allocate data structures
  size_t num_matched_points = match.size();
  map_points.reserve(num_matched_points);
  scan_points.reserve(num_matched_points);
  // for each match ( we dont need to transform all the map )
  for (const DMatch& correspondence : match) {
    const auto& mp = _mapPoint[correspondence.queryIdx];
    const auto& sp = kd2.keypoints[correspondence.trainIdx];
    // tg transform map point in previous frame
    Vector2f map_point = _motion.R.transpose() * (Vector2f(mp.pt.x, mp.pt.y) - _motion.t);
    // tg add points to data structures
    map_points.emplace_back(map_point);
    scan_points.emplace_back(Vector2f(sp.pt.x,sp.pt.y));
  }
  // tg compute mean point scan
  Vector2f mean_scan = Vector2f::Zero();
  for(const Vector2f& p : scan_points){
    mean_scan += p;
  }
  mean_scan /= static_cast<float>(num_matched_points);
  // tg compute mean point map
  Vector2f mean_map = Vector2f::Zero();
  for(const Vector2f& p : map_points){
    mean_map += p;
  }
  mean_map /= static_cast<float>(num_matched_points);
  // tg compute cross-correlation of point distribution
  Matrix2f sigma = Matrix2f::Zero();
  for(size_t i=0; i<num_matched_points; ++i){
    sigma += (map_points[i] - mean_map) * (scan_points[i] - mean_scan).transpose();
  }
  sigma /= static_cast<float>(num_matched_points);
  // recondition rotation matrix
  Eigen::JacobiSVD<Eigen::Matrix2f> svd(sigma, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Matrix2f R = svd.matrixU() * svd.matrixV().transpose();
  
  std::cerr << "   ^^^    " << '\n';
  std::cerr << " Rotation " << R <<'\n';
  // tg compute translation
  Eigen::Vector2f t = mean_map - R * mean_scan;
  std::cerr << "+++ Translation +++ "<< t << '\n';
  _highTranslation = t;

  // Dont go further if there is essentially no motion
  if (t.norm() < 0.2) {
    std::cerr << "No MoTion" << '\n';
    // tg mi fa sanguinare il culo
    SE2 noMotion;
    noMotion.R.setIdentity();
    noMotion.t << 0,0;
    return noMotion;
  }
  // tg i put the inverse here to keep consistency outside of this function
  _scanMotion.R = R.transpose();
  _scanMotion.t = - R.transpose() * t;
  // tg vomito male per la struct SE2
  // now transform is from current frame to last frame
  _motion.R *= R;
  _motion.t += t;
  _totalMotion.R *= R;
  _totalMotion.t += t;

  std::cerr << " §§§§§§§§§§ " << '\n';
  std::cerr << "total rotation "<< _totalMotion.R << '\n';
  std::cerr << "total motion "<< _totalMotion.t << '\n';

  // aiuto dio ti prego uccidimi
  SE2 robMotion;
  robMotion.R = R;
  robMotion.t = t;
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
