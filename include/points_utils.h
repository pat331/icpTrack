#pragma once
#include "defs.h"

namespace pr {

  //! draws a set of 2d points on an image
  //! @param img: the (preallocated) dest image
  //! @param points: the array of points
  //! @param color: the color of the points
  //! @param radius: the size of the point in the image (in pixels)
  void drawPoints(RGBImage& img,
		  const Vector2fVector& points,
		  const cv::Scalar& color,
		  int radius);

  //! draws a set of correspondences between 2d points
  //! @param img: the dest image (preallocated)
  //! @param reference_image_points: the first vector of points
  //! @param current_image_points: the second vector of points
  //! @param correspondences: the array of correspondences.
  //!        if correspondences[k] = (i,j), a line between
  //!        reference_image_points[i] and current_image_points[j]
  //!        will be drawn
  //! @param color: the color
  void drawCorrespondences(RGBImage& img,
			   const Vector2fVector& reference_image_points,
			   const Vector2fVector& current_image_points,
			   const IntPairVector& correspondences,
			   cv::Scalar color=cv::Scalar(0,255,0));


}
