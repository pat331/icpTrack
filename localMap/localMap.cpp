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

#include "localMap.h"

LocalMap::LocalMap(){
  _originImage(0) = 400;
  _originImage(1) = 400;
}

void LocalMap::initFirstMap(const std::vector<KeyPoint>& keypoints, const Mat& descriptors){
  _mapPoint = keypoints; // Hold the map
  _mapPointDescriptors= descriptors;
  // Bring the origin to (0,0)
  for (size_t i = 0; i < _mapPoint.size(); i++) {
    keypoints1[i].pt.x = keypoints1[i].pt.x - _originImage(0);
    keypoints1[i].pt.y = keypoints1[i].pt.y - _originImage(1);
  }

}
