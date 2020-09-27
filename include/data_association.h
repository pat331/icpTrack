#include "defs.h"

namespace pr {

  void computeCorrespondences(IntPairVector& correspondences,
  				const Vector2fVector from_points,
  				const Vector2fVector to_points,
          const float& gating_thres,
          const float& lbf_thres);


  void assignCorrespondences(IntPairVector& correspondences,
          const Vector2fVector from_points,
          const Vector2fVector to_points,
          const float& gating_thres);

  void refineCorrespondences(IntPairVector& correspondences,
					const Vector2fVector from_points,
					const Vector2fVector to_points,
					const float& gating_thres,
					const float& lbf_thres,
					const Eigen::Isometry2f rough_transform);
}
