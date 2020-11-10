#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>
#include "dataAssociationSURF.h"
#include "defs.h"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

class LocalMap{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //! ctor
    LocalMap();

    void initFirstMap(const std::vector<KeyPoint>& keypoints, const Mat& descriptors);

    void dispMap();

    void trackLocalMap(const std::vector<KeyPoint>& keypointsFrame,
                       const Mat& descriptorsFrame,
                       const Eigen::Matrix<float, 2, 2>& RotationMatrix,
                       const Eigen::Vector2f& translationVector);

    void mergeMap(const std::vector<KeyPoint>& keyFrame,
                  const std::vector<DMatch>& matchesForMerge,
                  const Eigen::Matrix<float, 2, 2>& RotationMatrix,
                  const Eigen::Vector2f& translationVector);


  protected:

    std::vector<KeyPoint> _mapPoint;
    Mat _mapPointDescriptors;
    Eigen::Vector2f _originImage;



};
