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


Eigen::Matrix<float, 1, Eigen::Dynamic> greedyAlgorithm(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> pairwiseCompatibilities, Eigen::Matrix<float, 3, Eigen::Dynamic> matchProposal){

  int numberOfAssociation = pairwiseCompatibilities.cols();
  Eigen::EigenSolver<Eigen::MatrixXf> es;
  // Eigen::MatrixXf A = MatrixXf::Random(4,4);
  es.compute(pairwiseCompatibilities, /* computeEigenvectors = */ true);

  cout << "The eigenvalues of A are: " << es.eigenvalues().transpose() << endl;
  cout << "The eigenVectors of A are: " << es.eigenvectors().transpose() << endl;
  std::cerr << "the first value of the first eigenvector are "<< es.eigenvectors().col(0)[0].real() << '\n';
  std::cerr << "singolo autovalore "<< es.eigenvalues()[0] << '\n';

  float maxEigenvalue;
  int indexOfPrincipalVector;
  std::vector<int> unsearched (pairwiseCompatibilities.cols(),0);
  std::vector<int> m (pairwiseCompatibilities.cols(),0);
  // std::vector<int> solution (pairwiseCompatibilities.cols(),0);
  Eigen::Matrix<float, 1, Eigen::Dynamic> solution;
  solution.resize(pairwiseCompatibilities.cols());
  solution.setZero();

  int score = 0;

  // Search for the principal eigenvector
  for (int i = 0; i < pairwiseCompatibilities.cols(); i++) {
    if (i == 0) {
      maxEigenvalue = es.eigenvalues()[i].real();
      indexOfPrincipalVector = i;
      continue;
    }

    if (maxEigenvalue < es.eigenvalues()[i].real()) {
      maxEigenvalue = es.eigenvalues()[i].real();
      indexOfPrincipalVector = i;
    }
    // The principal eigenVector is the es.eigenvectors().col(indexOfPrincipalVector)
  }
  float m_g;
  int indexM;
  int numberUnsearched = 0;
  float checkScore;

  std::cerr << "max" << maxEigenvalue <<  '\n';
  std::cerr << "index max " << indexOfPrincipalVector << '\n';
  do {
    m_g = 0; // possibile errore nel caso un autovettore fosse degenere
    for (int j = 0; j < pairwiseCompatibilities.cols(); j++) {
      if (m[j]>0 || unsearched[j] > 0) {
        std::cerr << "unserached and m exption if " << '\n';
        continue;
      }
      if (j==0 && m[j]==0) {
        m_g = es.eigenvectors().col(indexOfPrincipalVector)[j].real();
        indexM = j;
        continue;
      }
      std::cerr << "m_g pow = "<< pow(m_g,2) << '\n';
      std::cerr << "comparazione pow  "<< pow(es.eigenvectors().col(indexOfPrincipalVector)[j].real(),2)<< '\n';
      if (pow(m_g,2) <= pow(es.eigenvectors().col(indexOfPrincipalVector)[j].real(),2)) {
        std::cerr << "entrato " << '\n';
        m_g = es.eigenvectors().col(indexOfPrincipalVector)[j].real();
        indexM = j;
      }
    }
    m[indexM]=1;
    unsearched[indexM]=1;
    solution(indexM)=1;
    numberUnsearched++;
    // check termination
    // checkScore = (solution.transpose()*pairwiseCompatibilities*solution)(0);
    // if (checkScore< score) {
    //   break;
    // }
    // adesso bisogna controllare quale associazione abbiamo preso
    // i = indexM, j = ? (valore da cercare in matchProposal)
    int associationToEliminate = matchProposal(1,indexM);
    std::cerr << "numberUnsearched "<< numberUnsearched << '\n';
    for (int k = 0; k < pairwiseCompatibilities.cols(); k++) {
      if (k == indexM) {
        continue;
      }
      std::cerr << "mathc proposal "<< matchProposal(1,k) << '\n';
      std::cerr << "associationToEliminate "<<associationToEliminate << '\n';
      if (matchProposal(1,k) == associationToEliminate) {
        std::cerr << "entro cazo" << '\n';
        unsearched[k]=1;
        numberUnsearched++;
      }
    }
  } while(numberUnsearched <= unsearched.size());
  std::cerr << "solution vector optimization " << solution << '\n';

  return solution;
}
////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> createPairwiseCompatibilities(VectorOfDescriptorVector descriptor1, VectorOfDescriptorVector descriptor2, Eigen::Matrix<float, 3, Eigen::Dynamic> unaryMatch){
    //  Number of possible pairs in n landMark n(n-1)/2
    // Prendere tutte le coppie di indici possibili
    // int numberOfLandmarks = descriptor1.size();
    int numberOfLandmarks = 15;
    // Inizializzo la matrice del compatibility score
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> pairwiseCompatibilities;
    pairwiseCompatibilities.resize(numberOfLandmarks, numberOfLandmarks);

    Eigen::Vector2i indexAssociatedLandMarki;
    Eigen::Vector2i indexAssociatedLandMarkj;
    float distanceInScan1, distanceInScan2;

    for (int i = 0; i < numberOfLandmarks; i++) { // landmark scan2 trasformati
      indexAssociatedLandMarki(0) = (int)unaryMatch(1,i); // landmark index in scan2 associated to  landmark i
      indexAssociatedLandMarki(1) = (int)unaryMatch(1,i); // in the unaryMatch vector the information about the associated landmark is in the second component

      for (int j = i; j < numberOfLandmarks; j++) {
        indexAssociatedLandMarkj(0) = (int)unaryMatch(1,j); // landmark index in scan2 associated to  landmark j
        indexAssociatedLandMarkj(1) = (int)unaryMatch(1,j);

        distanceInScan1 = sqrt(pow(descriptor1[i](0)-descriptor1[j](0),2)
                              +pow(descriptor1[i](1)-descriptor1[j](1),2));
        distanceInScan2 = sqrt(pow(descriptor2[indexAssociatedLandMarki(0)](0)-descriptor2[indexAssociatedLandMarkj(0)](0),2)
                              +pow(descriptor2[indexAssociatedLandMarki(1)](1)-descriptor2[indexAssociatedLandMarkj(1)](1),2));

        pairwiseCompatibilities(i,j)= 1/(1+abs(distanceInScan1-distanceInScan2));
        pairwiseCompatibilities(j,i)= pairwiseCompatibilities(i,j);
      }
    }
    // Eigen::EigenSolver<Eigen::MatrixXf> es;
    //
    // es.compute(pairwiseCompatibilities, /* computeEigenvectors = */ true);
    // cout << "The eigenvalues of A are: " << es.eigenvalues().transpose() << endl;
    // cout << "The eigenvectors of A are: " << es.eigenvectors().transpose() << endl;
    // std::cerr << "prova eigen "<< es.eigenvalues().at(1) << '\n';
    return pairwiseCompatibilities;
}

Vector2fVector getPairInScan(int numberOfLandmarks){

  Vector2fVector vectorPair;
  Eigen::Vector2f pair;
  for (int i = 0; i < numberOfLandmarks-1; i++) {
    int scarto = i+1;
    std::cerr << "scarto "<< scarto << '\n';
    for (int j = scarto; j < numberOfLandmarks; j++) {
      pair(0)=i;
      pair(1)=j;
      vectorPair.push_back(pair);
    }
  }
  std::cerr << "prova errore su indici 1 inside "<< vectorPair[0](0) << '\n';
  std::cerr << "prova errore su indici 2 inside "<< vectorPair[0](1) << '\n';
  return vectorPair;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<float, 3, Eigen::Dynamic> matchProposal(VectorOfDescriptorVector descriptorScan1, VectorOfDescriptorVector descriptorScan2){
  // BRUTEFORCE
  float descriptorDistance;
  float bestDescriptorDistance;
  int indexAssociatedLandMark;

  const int dimDescriptorScan1 = (int)descriptorScan1.size();
  Eigen::Matrix<float, 3, Eigen::Dynamic> landMarkInScanAlreadyAssociated;
  landMarkInScanAlreadyAssociated.resize(3,dimDescriptorScan1);
  landMarkInScanAlreadyAssociated.setZero();

  for (size_t i = 0; i < descriptorScan1.size(); i++) {
    bestDescriptorDistance = 99999; // fisso  ad un valore alto per comodita'. Verra' immediatamente cambiato nella prima iterazione su j
    for (size_t j = 0; j < descriptorScan2.size(); j++) {
      descriptorDistance = 0;
      for (int k = 0; k < 400+3500; k++) { // Ricorda sempre che le prime due posizioni del descriptor sono occupate dalla posizione del landmark
        // descriptorDistance += abs( descriptorScan1[i](k+2) - descriptorScan2[j](k+2) );
        descriptorDistance += abs( descriptorScan1[i](k+2) - descriptorScan2[j](k+2) );
      }
      if (j==0 && i==0) {
        bestDescriptorDistance = descriptorDistance;
        indexAssociatedLandMark = 0;
      }
      if (descriptorDistance < bestDescriptorDistance) {
        bestDescriptorDistance = descriptorDistance;
        indexAssociatedLandMark = j; // Numero del landmark nello scan2 che stiamo associando al landmark i nello scan1
      }
    }

    landMarkInScanAlreadyAssociated(0,i) = 1; // Indicazione che il landmark e' stato associato
    landMarkInScanAlreadyAssociated(1,i) = indexAssociatedLandMark;  // Indicazione su quale landmark dello scan2 lo associamo
    landMarkInScanAlreadyAssociated(2,i) = bestDescriptorDistance; // Indicazione sulla distanza fra i due landmark

  }
  //////////////////////////////////////////////////////////////////////////////
  // for (size_t i = 0; i < descriptorScan2.size(); i++) {
  //   bestDescriptorDistance = 99999; // fisso  ad un valore alto per comodita'. Verra' immediatamente cambiato nella prima iterazione su j
  //   for (size_t j = 0; j < descriptorScan1.size(); j++) {
  //     descriptorDistance = 0;
  //     for (int k = 0; k < 400+3500; k++) { // Ricorda sempre che le prime due posizioni del descriptor sono occupate dalla posizione del landmark
  //       // descriptorDistance += abs( descriptorScan1[i](k+2) - descriptorScan2[j](k+2) );
  //       descriptorDistance += abs( descriptorScan2[i](k+2) - descriptorScan1[j](k+2) );
  //     }
  //     if (j==0 && i==0) {
  //       bestDescriptorDistance = descriptorDistance;
  //       indexAssociatedLandMark = 0;
  //     }
  //     if (descriptorDistance < bestDescriptorDistance) {
  //       bestDescriptorDistance = descriptorDistance;
  //       indexAssociatedLandMark = j; // Numero del landmark nello scan1 che stiamo associando al landmark i nello scan2
  //     }
  //   }
  //   // Landmark non ancora associato do -->
  //   if (landMarkInScanAlreadyAssociated(0,indexAssociatedLandMark) == 0) {
  //       landMarkInScanAlreadyAssociated(0,indexAssociatedLandMark) = 1; // Indicazione che il landmark e' stato associato
  //       landMarkInScanAlreadyAssociated(1,indexAssociatedLandMark) = i;  // Indicazione su quale landmark dello scan2 lo associamo
  //       landMarkInScanAlreadyAssociated(2,indexAssociatedLandMark) = bestDescriptorDistance; // Indicazione sulla distanza fra i due landmark
  //   } else if(landMarkInScanAlreadyAssociated(0,indexAssociatedLandMark) == 1){ // Il landmark j ha gia' una associazione
  //     // Controllo se questa associazione e' migliore
  //     if (bestDescriptorDistance < landMarkInScanAlreadyAssociated(2,indexAssociatedLandMark)) {
  //       landMarkInScanAlreadyAssociated(1,indexAssociatedLandMark) = i;  // Riassegno il landmark associato i
  //       landMarkInScanAlreadyAssociated(2,indexAssociatedLandMark) = bestDescriptorDistance; // Riassegno la distanza fra i due landmark associati
  //     }
  //   }
  //
  //
  // }
  //////////////////////////////////////////////////////////////////////////////
  // Eigen::Matrix<float, 1, Eigen::Dynamic> associationVector;
  // associationVector.resize(1,dimDescriptorScan1);
  // associationVector = landMarkInScanAlreadyAssociated.col(1);
  //
  // return associationVector;
  return landMarkInScanAlreadyAssociated;
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
