#include "data_association.h"

namespace pr {

	void computeCorrespondences(IntPairVector& correspondences,
					const Vector2fVector from_points,
					const Vector2fVector to_points,
					const float& gating_thres,
					const float& lbf_thres){
	  correspondences.resize(to_points.size());
	  int num_correspondences=0;
		// io non ho mai lo stesso numero di punti ----> come rimediare a questo fatto?
	  // assert(from_points.size()==to_points.size());

		// A logica mi sembrebbe meglio avere un numero di corrispondenze uguale al numero di landmark
		// dello scan che ne contiene di meno.
		// Questo è il motivo di questo if osceno

		if(to_points.size() <= from_points.size()){

			for (size_t to_iter=0; to_iter<to_points.size(); to_iter++){

				const Eigen::Vector2f& current_point=to_points[to_iter];
				float min_dist = 50;
				float second_min_dist = 50;
				int closest_index = -1;
				for (size_t from_iter=0; from_iter<from_points.size(); from_iter++){
					const Eigen::Vector2f& reference_point=from_points[from_iter];
					float dist = (current_point-reference_point).norm();
					if (dist < min_dist){
						closest_index = from_iter;
						second_min_dist = min_dist;
						min_dist = dist;
					}
					else if (dist < second_min_dist){
						second_min_dist = dist;
					}
				}
				// No correspondence and gating
				if ((closest_index == -1) || (min_dist > gating_thres) ){
					continue;
				}
				// std::cout << "dx " << from_points[closest_index] - current_point << std::endl;
				// Lonely besties 1
				// if (second_min_dist-min_dist < lbf_thres){
				// 	continue;
				// }
				// Best friends
				// float bf_min_dist = 10;
				// float bf_second_min_dist = 10;
				// int bf_closest_index = -1;
				// const Eigen::Vector2f& check_point=from_points[closest_index];
				//
				// for (size_t j=0; j<to_points.size(); j++){
				//
				// 	const Eigen::Vector2f& reference_point=to_points[j];
				// 	float dist = (check_point-reference_point).norm();
				// 	if (dist < bf_min_dist){
				// 		bf_closest_index = j;
				// 		bf_second_min_dist = bf_min_dist;
				// 		bf_min_dist = dist;
				// 	}
				// 	else if (dist < second_min_dist){
				// 		bf_second_min_dist = dist;
				// 	}
				// }
				// if (bf_closest_index != to_iter){
				// 	continue;
				// }
				// //Lonely besties 2
				// if (bf_second_min_dist-bf_min_dist < lbf_thres){
				// 	continue;
				// }

				IntPair& correspondence=correspondences[num_correspondences];
				correspondence.first=to_iter;
				correspondence.second=closest_index;

				// std::cout << correspondence.first << ", " << correspondence.second << std::endl;

				num_correspondences++;
			}

		}

		else{
			
			for (size_t from_iter=0; from_iter<from_points.size(); from_iter++){

				const Eigen::Vector2f& current_point=from_points[from_iter];
				float min_dist = 50;
				float second_min_dist = 50;
				int closest_index = -1;
				for (size_t to_iter=0; to_iter<to_points.size(); to_iter++){
					const Eigen::Vector2f& reference_point=to_points[to_iter];
					float dist = (current_point-reference_point).norm();
					if (dist < min_dist){
						closest_index = to_iter;
						second_min_dist = min_dist;
						min_dist = dist;
					}
					else if (dist < second_min_dist){
						second_min_dist = dist;
					}
				}
				// No correspondence and gating
				if ((closest_index == -1) || (min_dist > 3) ){
					continue;
				}
				// std::cout << "dx " << from_points[closest_index] - current_point << std::endl;
				// Lonely besties 1
				// if (second_min_dist-min_dist < lbf_thres){
				// 	continue;
				// }

				////////////////////////////////////////////////////////////////////////
				// Best friends
				// float bf_min_dist = 30;
				// float bf_second_min_dist = 30;
				// int bf_closest_index = -1;
				// const Eigen::Vector2f& check_point=to_points[closest_index];
				//
				// for (size_t j=0; j<from_points.size(); j++){
				//
				// 	const Eigen::Vector2f& reference_point=from_points[j];
				// 	float dist = (check_point-reference_point).norm();
				// 	if (dist < bf_min_dist){
				// 		bf_closest_index = j;
				// 		bf_second_min_dist = bf_min_dist;
				// 		bf_min_dist = dist;
				// 	}
				// 	else if (dist < second_min_dist){
				// 		bf_second_min_dist = dist;
				// 	}
				// }
				// if (bf_closest_index != from_iter){
				// 	continue;
				// }
				////////////////////////////////////////////////////////////////////////
				// //Lonely besties 2
				// if (bf_second_min_dist-bf_min_dist < lbf_thres){
				// 	continue;
				// }

				IntPair& correspondence=correspondences[num_correspondences];
				correspondence.first=closest_index;
				correspondence.second=from_iter;

				// std::cout << correspondence.first << ", " << correspondence.second << std::endl;

				num_correspondences++;
			}

		}


		correspondences.resize(num_correspondences);

	}

	void refineCorrespondences(IntPairVector& correspondences,
					const Vector2fVector from_points,
					const Vector2fVector to_points,
					const float& gating_thres,
					const float& lbf_thres,
					const Eigen::Isometry2f rough_transform){
		correspondences.resize(to_points.size());
		int num_correspondences=0;
		assert(from_points.size()==to_points.size());

		for (size_t to_iter=0; to_iter<to_points.size(); to_iter++){
		// for (size_t to_iter=200; to_iter<230; to_iter++){
			const Eigen::Vector2f& current_point=to_points[to_iter];
			float min_dist = 1;
			float second_min_dist = 1;
			int closest_index = -1;
			for (size_t from_iter=0; from_iter<from_points.size(); from_iter++){
				const Eigen::Vector2f& reference_point=from_points[from_iter];
				float dist = (current_point-reference_point).norm();
				if (dist < min_dist){
					closest_index = from_iter;
					second_min_dist = min_dist;
					min_dist = dist;
				}
				else if (dist < second_min_dist){
					second_min_dist = dist;
				}
			}
			// No correspondence and gating
			if ((closest_index == -1) || (min_dist > gating_thres) ){
				continue;
			}

			// Discard correspondences that don't support first estimate of transform
			Eigen::Vector2f dx = current_point - from_points[closest_index];
			Eigen::Vector2f dx_after_estim = current_point - rough_transform*from_points[closest_index];

			if(dx.norm() < dx_after_estim.norm()){
				continue;
			}

			// Lonely besties 1
			// if (second_min_dist-min_dist < lbf_thres){
			// 	continue;
			// }
			// Best friends
			float bf_min_dist = 1;
			float bf_second_min_dist = 1;
			int bf_closest_index = -1;
			const Eigen::Vector2f& check_point=from_points[closest_index];

			for (size_t j=0; j<to_points.size(); j++){

				const Eigen::Vector2f& reference_point=to_points[j];
				float dist = (check_point-reference_point).norm();
				if (dist < bf_min_dist){
					bf_closest_index = j;
					bf_second_min_dist = bf_min_dist;
					bf_min_dist = dist;
				}
				else if (dist < second_min_dist){
					bf_second_min_dist = dist;
				}
			}
			if (bf_closest_index != to_iter){
				continue;
			}
			// //Lonely besties 2
			// if (bf_second_min_dist-bf_min_dist < lbf_thres){
			// 	continue;
			// }

			IntPair& correspondence=correspondences[num_correspondences];
			correspondence.first=to_iter;
			correspondence.second=closest_index;

			// std::cout << correspondence.first << ", " << correspondence.second << std::endl;

			num_correspondences++;
		}
		correspondences.resize(num_correspondences);

	}
}
