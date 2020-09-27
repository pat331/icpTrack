#include "smicp_solver.h"

#include <Eigen/Cholesky>
#include <iostream>

namespace pr {

  SMICPSolver::SMICPSolver(){
    _from_points=0;
    _to_points=0;
    _damping=1;
    _min_num_inliers=0;
    _num_inliers=0;
    _kernel_thereshold=1000; // 33 pixels
  }

  void SMICPSolver::init(const Eigen::Isometry2f& transform,
                        const Vector2fVector& from_points,
                        const Vector2fVector& to_points){
    _transform = transform;
    _from_points=&from_points;
    _to_points=&to_points;
  }


  bool SMICPSolver::errorAndJacobian(Eigen::Vector2f& error,
                                    Matrix2_3f& jacobian,
                                    const Eigen::Vector2f& from_point,
                                    const Eigen::Vector2f& to_point){
    // compute the prediction
    Eigen::Vector2f predicted_point = _transform * from_point;

    error=predicted_point-to_point;

    // std::cout << "error: \n" << error << std::endl;

    // compute the jacobian of the transformation
    jacobian=Eigen::Matrix<float, 2,3>::Zero();
    jacobian.block<2,2>(0,0).setIdentity();
    Eigen::Matrix<float, 2,1> dP;
    dP << -predicted_point(1),
          predicted_point(0);
    jacobian.block<2,1>(0,2)=dP;
    // std::cout << "predicted point \n";
    // std::cout << predicted_point << std::endl;
    // std::cout << jacobian << std::endl;

    return true;
  }

  void SMICPSolver::linearize(const IntPairVector& correspondences, bool keep_outliers){
    _H.setZero();
    _b.setZero();
    _num_inliers=0;
    _chi_inliers=0;
    _chi_outliers=0;
    for (const IntPair& correspondence: correspondences){
      Eigen::Vector2f e;
      Matrix2_3f J;
      int ref_idx=correspondence.first;
      int curr_idx=correspondence.second;
      bool inside=errorAndJacobian(e,
                                   J,
                                   (*_from_points)[curr_idx],
                                   (*_to_points)[ref_idx]);
      if (! inside)
        continue;

      float chi=e.dot(e);

      _H+=J.transpose()*J;
      _b+=J.transpose()*e;
    }
  }

  bool SMICPSolver::oneRound(const IntPairVector& correspondences, bool keep_outliers){
    using namespace std;
    linearize(correspondences, keep_outliers);

    //compute a solution
    Eigen::Vector3f dx = _H.ldlt().solve(-_b);

    _transform = v2t(dx)*_transform;
    
    return true;
  }
}
