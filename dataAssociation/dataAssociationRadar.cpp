#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>

#include "dataAssociationRadar.h"
#include "defs.h"


using namespace std;
using namespace cv;
using namespace pr;


void createPairwiseCompatibilities(VectorOfDescriptorVector descriptor1, VectorOfDescriptorVector descriptor2){

}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<float, 3, Eigen::Dynamic> matchProposal(VectorOfDescriptorVector descriptorScan1, VectorOfDescriptorVector descriptorScan2){
  // BRUTEFORCE
  float descriptorDistance;
  float bestDescriptorDistance;
  int indexAssociatedLandMark;

  const int dimDescriptorScan2 = (int)descriptorScan2.size();
  Eigen::Matrix<float, 3, Eigen::Dynamic> landMarkInScan2AlreadyAssociated;
  landMarkInScan2AlreadyAssociated.resize(3,dimDescriptorScan2);
  landMarkInScan2AlreadyAssociated.setZero();

  for (size_t i = 0; i < descriptorScan2.size(); i++) {
    bestDescriptorDistance = 99999; // fisso  ad un valore alto per comodita'. Verra' immediatamente cambiato nella prima iterazione su j
    for (size_t j = 0; j < descriptorScan1.size(); j++) {
      descriptorDistance = 0;
      for (int k = 0; k < 400+3500; k++) { // Ricorda sempre che le prime due posizioni del descriptor sono occupate dalla posizione del landmark
        // descriptorDistance += abs( descriptorScan1[i](k+2) - descriptorScan2[j](k+2) );
        descriptorDistance += abs( descriptorScan2[i](k+2) - descriptorScan1[j](k+2) );
      }
      if (j==0 && i==0) {
        bestDescriptorDistance = descriptorDistance;
        indexAssociatedLandMark = 0;
      }
      if (descriptorDistance < bestDescriptorDistance) {
        bestDescriptorDistance = descriptorDistance;
        indexAssociatedLandMark = j; // Numero del landmark nello scan1 che stiamo associando al landmark i nello scan2
      }
    }
    // Landmark non ancora associato do -->
    if (landMarkInScan2AlreadyAssociated(0,indexAssociatedLandMark) == 0) {
        landMarkInScan2AlreadyAssociated(0,indexAssociatedLandMark) = 1; // Indicazione che il landmark e' stato associato
        landMarkInScan2AlreadyAssociated(1,indexAssociatedLandMark) = i;  // Indicazione su quale landmark dello scan2 lo associamo
        landMarkInScan2AlreadyAssociated(2,indexAssociatedLandMark) = bestDescriptorDistance; // Indicazione sulla distanza fra i due landmark
    } else if(landMarkInScan2AlreadyAssociated(0,indexAssociatedLandMark) == 1){ // Il landmark j ha gia' una associazione
      // Controllo se questa associazione e' migliore
      if (bestDescriptorDistance < landMarkInScan2AlreadyAssociated(2,indexAssociatedLandMark)) {
        landMarkInScan2AlreadyAssociated(1,indexAssociatedLandMark) = i;  // Riassegno il landmark associato i
        landMarkInScan2AlreadyAssociated(2,indexAssociatedLandMark) = bestDescriptorDistance; // Riassegno la distanza fra i due landmark associati
      }
    }
  }

  return landMarkInScan2AlreadyAssociated;
}

////////////////////////////////////////////////////////////////////////////////
VectorOfDescriptorVector createDescriptor(Mat L){

  float theta, rho;

  Eigen::Vector2f positionLandMark;
  Vector2fVector landmarksPositionPolar;
  Vector2fVector landmarksPositionCart;
  Vector2fVector landmarksPositionPolarInFrameLandmark;

  VectorDescriptor descriptorCurrentLandmark;
  VectorOfDescriptorVector descriptorLandmarkInRadarScan;
  int positionHelper;

  std::vector<float> angularHistogram(400,0);
  std::vector<float> annulusHistogram(3500,0);

  int normAngle = 0;
  int normAnnulus = 0;

  landmarksPositionPolar = getLandMarkPolarCoord(L);

  for (int i = 0; i < landmarksPositionPolar.size(); i++) {
    landmarksPositionPolarInFrameLandmark = getLandMarkPolarCoordInLandMarkFrame(landmarksPositionPolar, i);
    std::fill(angularHistogram.begin(), angularHistogram.end(), 0);
    std::fill(annulusHistogram.begin(), annulusHistogram.end(), 0);

    for (int k = 0; k < landmarksPositionPolarInFrameLandmark.size(); k++) {

      angularHistogram[(int)landmarksPositionPolarInFrameLandmark[k](1)]++;
      annulusHistogram[(int)(landmarksPositionPolarInFrameLandmark[k](0)/2)]++;

    }
    // find the max number of element in one slice
    for (size_t k = 0; k < angularHistogram.size(); k++) {
      if (angularHistogram[k]>normAngle) {
        normAngle = angularHistogram[k];
      }
    }
    // fill the angular part of the descriptor
    for (size_t k = 0; k < angularHistogram.size(); k++) {
      descriptorCurrentLandmark[k+2] = angularHistogram[k]/normAngle; // Plus two because the first two position are occupied by the position of the landMark
    }
    // find the maximum number of element in one annulus
    for (size_t j = 0; j < annulusHistogram.size(); j++) {
      if (annulusHistogram[j]>normAnnulus) {
        normAnnulus = annulusHistogram[j];
      }
    }
    // fill the annulus part of the descriptor
    for (size_t j = 0; j < annulusHistogram.size(); j++) {
      descriptorCurrentLandmark[j+2+400] = annulusHistogram[j]/normAnnulus;
    }

    rho = landmarksPositionPolar[i](0);
    theta = landmarksPositionPolar[i](1)*angleResolution;
    descriptorCurrentLandmark[0] = rho*cos(theta);
    descriptorCurrentLandmark[1] = rho*sin(theta);

    descriptorLandmarkInRadarScan.push_back(descriptorCurrentLandmark);

  }
  return descriptorLandmarkInRadarScan;

}
////////////////////////////////////////////////////////////////////////////////
Vector2fVector getLandMarkPolarCoord(Mat L){
  std::cerr << "L size " << L.size() << '\n';
  Eigen::Vector2f positionLandMark;
  Vector2fVector positionPolar;

  for (int x = 0; x < L.cols; x++) {
    for (int y = 0; y < L.rows; y++) {
      if (L.at<float>(Point(x, y)) > 0) {  // Se il punto e' un landmark procediamo a calcolare il desciptor
        positionLandMark(0) = x;
        positionLandMark(1) = y;
        positionPolar.push_back(positionLandMark); // x = rho, y = theta
      }
    }
  }
  return positionPolar;
}

Vector2fVector getLandMarkPolarCoordInLandMarkFrame(Vector2fVector positionsLandMark, int landmark){
  Vector2fVector polarPositionsInFrameLand;
  Eigen::Vector2f positionLandMarkCartInFrame;
  Eigen::Vector2f positionLandMarkPolInFrame;
  Eigen::Vector2f newFrameCenter;

  float theta, rho;
  rho = positionsLandMark[landmark](0);
  theta = positionsLandMark[landmark](1)*angleResolution;
  newFrameCenter(0)= rho*cos(theta);
  newFrameCenter(1)= rho*sin(theta);

  for (int j = 0; j < positionsLandMark.size(); j++) {

    if (j==landmark) {
      continue;
    }
    rho = positionsLandMark[j](0);
    theta = positionsLandMark[j](1)*angleResolution;

    positionLandMarkCartInFrame(0) = rho*cos(theta) + newFrameCenter(0);
    positionLandMarkCartInFrame(1) = rho*sin(theta) + newFrameCenter(1);

    positionLandMarkPolInFrame(0) = sqrt(pow(positionLandMarkCartInFrame(0),2) + pow(positionLandMarkCartInFrame(1),2));
    positionLandMarkPolInFrame(1) = (float) atan2((double)positionLandMarkCartInFrame(1), (double)positionLandMarkCartInFrame(0));
    if (positionLandMarkPolInFrame(1)<0) {
      positionLandMarkPolInFrame(1) += 2*3.14;
    }
    positionLandMarkPolInFrame(1) = positionLandMarkPolInFrame(1)/angleResolution;
    positionLandMarkPolInFrame(1) = floor(positionLandMarkPolInFrame(1));
    polarPositionsInFrameLand.push_back(positionLandMarkPolInFrame);
  }

  return polarPositionsInFrameLand;
}
