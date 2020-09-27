#pragma once
#include "defs.h"

namespace pr {

  /**
     Solver for scan-matching problem.
     - create an object
     - initialize it passing:
       - the image points (that represent the measurements)
       - the world points (that represent the model)
       - A camera whose pose is initialized at initial_guess
     - call oneRound(<correspondences>) a bunch of times, with the correspondences returned by the finder;
       at each call, the solution will be subject to one ls operation
   */
  class SMICPSolver{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //! ctor
    SMICPSolver();

    //! init method, call it at the beginning
    //! @param camera: the camera
    //! @param world_points: the points of the world
    //! @param image_points: the points of the reference
    void init(const Eigen::Isometry2f& transform,
                          const Vector2fVector& from_points,
                          const Vector2fVector& to_points);

    const Eigen::Isometry2f& transform() const {return _transform;}

    //! chi square of the "good" points
    const float chiInliers() const {return _chi_inliers;}

    //! chi square of the "bad" points
    const float chiOutliers() const {return _chi_outliers;}

    //! number of inliers (an inlier is a point whose error is below kernel threshold)
    const int numInliers() const {return _num_inliers;}

    //! performs one iteration of optimization
    //! @param correspondences: the correspondences (first: measurement, second:model);
    //! param keep_outliers: if true, the outliers are considered in the optimization
    //! (but cut by the kernel)
    bool oneRound(const IntPairVector& correspondences, bool keep_outliers);

  protected:

    bool errorAndJacobian(Eigen::Vector2f& error,
                                      Matrix2_3f& jacobian,
                                      const Eigen::Vector2f& from_point,
                                      const Eigen::Vector2f& to_point);

    void linearize(const IntPairVector& correspondences, bool keep_outliers);


    Eigen::Isometry2f _transform;                  //< this will hold our state
    float _kernel_thereshold;        //< threshold for the kernel
    float _damping;                  //< damping, to slow the solution
    int _min_num_inliers;            //< if less inliers than this value, the solver stops
    const Vector2fVector* _from_points;
    const Vector2fVector* _to_points;
    Eigen::Matrix3f _H;
    Eigen::Vector3f _b;
    float _chi_inliers;
    float _chi_outliers;
    int _num_inliers;
  };

}
