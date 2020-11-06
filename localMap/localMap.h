#include "defs.h"

class LocalMap{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //! ctor
    LocalMap();

    void initFirstMap(const std::vector<KeyPoint>& keypoints, const Mat& descriptors);


  protected:

    std::vector<KeyPoint> _mapPoint;
    Mat _mapPointDescriptors;
    Eigen::Vector2f _originImage;



}
