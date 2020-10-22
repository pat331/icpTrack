#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>

#include "rigidBodyMotion.h"
#include "singleValueDecomposition2D.h"
#include "defs.h"


using namespace std;
using namespace cv;
using namespace pr;

Eigen::Matrix<float, 2, 2> rigidBodyMotion(VectorOfDescriptorVector &descriptor1,
                                           VectorOfDescriptorVector &descriptor2,
                                           Eigen::Matrix<float, 3, Eigen::Dynamic> &matchProposal,
                                           Eigen::Matrix<float, 1, Eigen::Dynamic> optimizedAssociationSolution){

  Eigen::Vector2f meanFirstScan = meanScan(descriptor1);
  std::cerr << "meanFirstScan "<< meanFirstScan << '\n';
  Eigen::Vector2f meanSecondScan = meanScan(descriptor2);
  std::cerr << "meanSecondScan "<< meanSecondScan << '\n';

  Vector2fVector xPrimeVector = positionPrime(descriptor1, meanFirstScan);
  Vector2fVector yPrimeVector = positionPrime(descriptor2, meanSecondScan);

  // Ricorda che la cross crossCorrelationMatrix va fatta solo delle associazioni approvate dal greedyAlgorithm
  Eigen::Matrix<float, 2, 2> crossCorrelationMatrix = computeCrossCorrelationMatrix(xPrimeVector,
                                                                                    yPrimeVector,
                                                                                    matchProposal,
                                                                                    optimizedAssociationSolution);
  std::cerr << "crossCorrelationMatrix "<< crossCorrelationMatrix << '\n';
  //Computation of singularValueDecomposition2D
  Eigen::Matrix<float, 2, 2> U = computationOfU(crossCorrelationMatrix(0,0),
                                                crossCorrelationMatrix(0,1),
                                                crossCorrelationMatrix(1,0),
                                                crossCorrelationMatrix(1,1));

  Eigen::Matrix<float, 2, 2> V = computationOfV(crossCorrelationMatrix(0,0),
                                                crossCorrelationMatrix(0,1),
                                                crossCorrelationMatrix(1,0),
                                                crossCorrelationMatrix(1,1));

  // Eigen::Matrix<float, 2, 2> R = U * V.transpose();

  //////////////////////////////////////////////////////////////////////////////
  // PROVA CON EIGEN SVD
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(crossCorrelationMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  // cout << "Its singular values are:" << endl << svd.singularValues() << endl;
  cout << "Its left singular vectors are the columns of the thin U matrix:" << endl << svd.matrixU() << endl;
  cout << "Its right singular vectors are the columns of the thin V matrix:" << endl << svd.matrixV() << endl;
  //////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix<float, 2, 2> R = svd.matrixU() * svd.matrixV().transpose();
  return R;
                                       }

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<float, 2, 2> computeCrossCorrelationMatrix(Vector2fVector &xPrime,
                                                         Vector2fVector &yPrime,
                                                         Eigen::Matrix<float, 3, Eigen::Dynamic> &matchProposal,
                                                         Eigen::Matrix<float, 1, Eigen::Dynamic> optimizedAssociationSolution){
  Eigen::Matrix<float, 2, 2> crossCorrelationMatrix;
  Eigen::Matrix<float, 2, 2> partialComputation;
  crossCorrelationMatrix.setZero();
  int indexAssiociated;
  float N = (float)xPrime.size();
  for (size_t i = 0; i < xPrime.size(); i++) {
    indexAssiociated = (int) matchProposal(1,i);
    if (optimizedAssociationSolution(i) == 1) {
      partialComputation = yPrime[indexAssiociated]*xPrime[i].transpose();
      // partialComputation = xPrime[i]*yPrime[indexAssiociated].transpose();
      partialComputation = partialComputation*1/N;

      crossCorrelationMatrix(0,0) += partialComputation(0,0);
      crossCorrelationMatrix(0,1) += partialComputation(0,1);
      crossCorrelationMatrix(1,0) += partialComputation(1,0);
      crossCorrelationMatrix(1,1) += partialComputation(1,1);

    }

  }
  return crossCorrelationMatrix;
}


Eigen::Vector2f meanScan(VectorOfDescriptorVector &descriptor){
  // int numberOfLandmarks = descriptor.size();
  float sumPositionX = 0;
  float sumPositionY = 0;
  for (size_t i = 0; i < descriptor.size(); i++) {
    sumPositionX += descriptor[i](0);
    sumPositionY += descriptor[i](1);
  }
  std::cerr << "sumPositionX "<< sumPositionX << '\n';
  std::cerr << "sumPositionY "<< sumPositionY << '\n';
  std::cerr << "descriptor size "<< descriptor.size() << '\n';
  sumPositionX = sumPositionX / (int) descriptor.size();
  sumPositionY = sumPositionY / (int) descriptor.size();
  // sumPositionX = sumPositionX/divisor;
  // sumPositionY = sumPositionY/divisor;

  Eigen::Vector2f meanScan;
  meanScan << sumPositionX,sumPositionY;
  return meanScan;
}

Vector2fVector positionPrime(VectorOfDescriptorVector &descriptor, Eigen::Vector2f meanScan){
    Vector2fVector prime;
    Eigen::Vector2f originalPosition;
    for (size_t i = 0; i < descriptor.size(); i++) {
      originalPosition << descriptor[i](0) - meanScan(0), descriptor[i](1) - meanScan(1);
      prime.push_back(originalPosition);
    }
    return prime;
}
