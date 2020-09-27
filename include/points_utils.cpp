#include <iostream>
#include "points_utils.h"

namespace pr {
  using namespace std;

  void drawPoints(RGBImage& img,
                  const Vector2fVector& points,
                  const cv::Scalar& color,
                  int radius){
    int rows=img.rows;
    int cols=img.cols;
    for (const Eigen::Vector2f point: points){
      int r=point.y();
      int c=point.x();
      if(r<0||r>=rows)
        continue;
      if(c<0||c>=cols)
        continue;
      cv::circle(img, cv::Point(c,r), radius, color);
    }
  }

  void drawCorrespondences(RGBImage& img,
                           const Vector2fVector& reference_image_points,
                           const Vector2fVector& current_image_points,
                           const IntPairVector& correspondences,
                           cv::Scalar color){
    for (const IntPair& correspondence: correspondences) {
      int ref_idx=correspondence.second;
      int curr_idx=correspondence.first;
      const Eigen::Vector2f& reference_point=reference_image_points[ref_idx];
      const Eigen::Vector2f& current_point=current_image_points[curr_idx];
      cv::line(img,
               cv::Point(reference_point.x(), reference_point.y()),
               cv::Point(current_point.x(), current_point.y()),
               color);
    }
  }


}
