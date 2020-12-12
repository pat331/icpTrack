#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>
#include "dataAssociationSURF.h"
#include "points_utils.h"
#include "rigidBodyMotion.h"
#include "defs.h"
#include "smicp_solver.h"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace pr;

class Mapping{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //! ctor
    Mapping();

    void initMap(const KeyAndDesc& kd, const SE2& mappingMotion);

    void dispMotion();

    KeyAndDesc findScanKeyPoint(const Mat& scan);

    std::vector<DMatch> matchMap(const KeyAndDesc& kd);

    SE2 scanMap(const KeyAndDesc& kd2,
                const std::vector<DMatch>& match);

    void robotMotion();

    void insertKeyFrame(const KeyAndDesc& kd2);

    void insertFrameReset(const KeyAndDesc& kd2, const SE2& motion);

    bool resetMap();

    void clearMap();

    void errorEstimation(const Eigen::Isometry2f& isometry_gt,
                         const Eigen::Isometry2f& isometry_calc);
    void dispError();


  protected:

    std::vector<KeyPoint> _mapPoint;
    Mat _mapDescriptors;

    SE2 _motion;
    SE2 _totalMotion;
    SE2 _scanMotion;

    SE2 _total;

    Vector2fVector _robotPose;
    Eigen::Vector2f _pose;

    int _tooMuchPoint;
    Eigen::Vector2f _highTranslation;


    double _totalErrorTranslation;
    double _totalErrorRotation;

};
