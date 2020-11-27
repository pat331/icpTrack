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
    void dispMotion();

    std::vector<DMatch> matchingWithMap(const std::vector<KeyPoint>& keypointsFrame,
                                        const Mat& descriptorsFrame);

    SE2 trackLocalMap(const std::vector<KeyPoint>& keypointsFrame1,
                       const Mat& descriptorsFrame1,
                       const std::vector<KeyPoint>& keypointsFrame,
                       const Mat& descriptorsFrame,
                       const SE2 scanMotion,
                       const std::vector<DMatch>& matchWithMap,
                       const std::vector<int>& associatedLandmarkIndex);

    void mergeMap(const std::vector<KeyPoint>& keyFrame,
                  const std::vector<DMatch>& matchesForMerge,
                  const Eigen::Matrix<float, 2, 2>& RotationMatrix,
                  const Eigen::Vector2f& translationVector);

    void insertKeyFrame(const std::vector<KeyPoint>& keyFrame,
                        const Mat& descriptorsFrame,
                        const Eigen::Matrix<float, 2, 2>& R,
                        const Eigen::Vector2f& t);

    void insertKeyFrame2(const std::vector<KeyPoint>& keyFrame,
                        const Mat& descriptorsFrame,
                        const Eigen::Matrix<float, 2, 2>& R,
                        const Eigen::Vector2f& t);

    void robotMotion(const SE2& robMotion);

    void fillNKS(int numberKeyPoints);

    void errorEstimation();


  protected:

    std::vector<KeyPoint> _mapPoint;
    Mat _mapPointDescriptors;
    std::vector<KeyPoint> _partialMap;
    Mat _partialDescriptors;
    std::vector<int> _mapPointAssociated;

    Eigen::Vector2f _originImage;
    int numeroDescrittori;
    int numeroScan;
    Vector2fVector _robotPose;


    std::vector<int> _numberKeyPointsInScan;




};
