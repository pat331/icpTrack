#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>

#include "dataAssociationSURF.h"
#include "defs.h"


using namespace std;
using namespace cv;
using namespace pr;

std::vector<int> createConsistencyMatrix(const vector<KeyPoint>& keypoints1,
                                         const vector<KeyPoint>& keypoints2,
                                         const vector< DMatch >& matches){

  float deltaC = 2; // Can be much smaller
  float distanceOnScan1;
  float distanceOnScan2;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> consistencyMatrix;
  consistencyMatrix.resize(matches.size(),matches.size());
  consistencyMatrix.setZero();

  for (size_t i = 0; i < matches.size(); i++) {
    // Calculate the first part of the pairwise constraint
    distanceOnScan1 = sqrt(pow(keypoints1[matches[i].queryIdx].pt.x - keypoints2[matches[i].trainIdx].pt.x,2)
                          +pow(keypoints1[matches[i].queryIdx].pt.y - keypoints2[matches[i].trainIdx].pt.y,2));

    for (size_t j = 0; j < matches.size(); j++) {
      if(j==i){
        consistencyMatrix(i,j) = 1;
      }
      distanceOnScan2 = sqrt(pow(keypoints1[matches[j].queryIdx].pt.x - keypoints2[matches[j].trainIdx].pt.x,2)
                            +pow(keypoints1[matches[j].queryIdx].pt.y - keypoints2[matches[j].trainIdx].pt.y,2));
      if (abs(distanceOnScan1 - distanceOnScan2)< deltaC) {
        consistencyMatrix(i,j) = 1;
      }
    }
  }
  // std::cerr << "consistencyMatrix "<<consistencyMatrix << '\n';
  std::vector<int> clique;
  clique = Grasp(consistencyMatrix);
  for (size_t i = 0; i < clique.size(); i++) {

  }

  return clique;

}

std::vector<int> Grasp(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &consistencyMatrix){
  std::vector<int> Q(consistencyMatrix.cols(),0); // Initiale clique
  std::vector<int> C(consistencyMatrix.cols(),1); // Initially all the verteces are candidate.

  int finishLoop = consistencyMatrix.cols();
  int checkC = 0;
  float d, dMin, dMax;
  float alpha = 0.2;
  int randomIndex;
  int checkRCL = 0;

  while (checkC != finishLoop) {
    // Calculation of dMin and dMax
    dMin = consistencyMatrix.cols();
    dMax = 0;
    std::vector<int> grades(consistencyMatrix.cols(),0);
    for (size_t i = 0; i < C.size(); i++) {
      d = 0;
      if (C[i] == 1) {
        for (size_t j = 0; j < consistencyMatrix.cols(); j++) {
          d += consistencyMatrix(i,j);
        }
        grades[i] = d;
        if (d >= dMax) {
          dMax = d;
        }
        if (d <= dMin) {
          dMin = d;
        }
      }
    }

    //
    // Construction of RCL set
    std::vector<int> RCL(consistencyMatrix.cols(),0);
    for (size_t k = 0; k < C.size(); k++) {
      if (C[k] == 1 && grades[k] > (int)(dMin + alpha*(dMax - dMin))) {
        RCL[k] = 1;
      }
    }
    float prova =0;
    for (size_t k = 0; k < RCL.size(); k++) {
      prova += RCL[k];
    }
    if (prova == 0) {
      break;
    }
    //
    // Select a random node from RCL
    checkRCL = 0;

    while (checkRCL == 0) {
      randomIndex = rand() % RCL.size();
      if (RCL[randomIndex] == 1) {
        Q[randomIndex] = 1;
        checkRCL = 1;
        for (size_t i = 0; i < C.size(); i++) {
          if (consistencyMatrix(randomIndex,i) == 0) {
            C[i] = 0;
          } else if (i == randomIndex) {
            C[i] = 0;
          }
        }
      }
    }

    checkC = consistencyMatrix.cols();
    for (size_t u = 0; u < C.size(); u++) {
      checkC = checkC - C[u];
    }

    // Se C ha tutti zeri al suo interno
    // checkC = finishLoop

    // Return the max clique;

  }

    return Q;
}
