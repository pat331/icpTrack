#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>

#include "preProcessRadarScan.h"
#include "defs.h"

using namespace std;
using namespace cv;
using namespace pr;


void keyPointExtraction(Mat radarScanImage, int maxNumberKeyPoint){

  Mat HMatrix;
  Mat prewittImage;
  Mat SPrime;
  Mat cart;

  int rows = radarScanImage.rows;
  int cols = radarScanImage.cols;
  Mat_<float> R(rows,cols);
  R = Mat::zeros(rows, cols, CV_32FC1);
  Mat_<float> keyPoint(rows,cols);
  keyPoint = Mat::zeros(rows, cols, CV_32FC1);
  std::cout << "SIZE DI R  "<< R.size()  << std::endl;

  prewittImage = prewittOperator(radarScanImage);
  SPrime = getMatrixSPrime(radarScanImage);
  HMatrix = getMatrixH(prewittImage, SPrime);

  Vector3fVector indicesMatrixH = getIndicesOfElementsInDescendingOrder(HMatrix);

  int l=0;
  int iter=0;
  int noneInR;
  int anyFalseInR = 0;
  float a,r; // Paul Newman notation for angle e range
  float valuePixelSPrime, valuePixelR;
  Eigen::Vector2f rangeBoundaries;
  Vector3fVector markedRegion;
  Eigen::Vector3f slice;
  Vector2fVector finalMarkedRegion;

  do {

    r = indicesMatrixH[iter](0);
    a = indicesMatrixH[iter](1);
    noneInR = 0;

    if (R.at<float>(Point(r, a)) == 0) { // point(a,r) must not be already in the region R
      rangeBoundaries = findRangeBoundaries(a,r,SPrime);
      for (int i = (int) rangeBoundaries(0); i <= (int) rangeBoundaries(1); i++) {
        if (R.at<float>(Point(i, a)) == 1) {
          // not a new region --> no increment of value l
          noneInR = 1;

          for (size_t k = 0; k < markedRegion.size(); k++) {
            if (markedRegion[k](0) == a) {

              if (markedRegion[k](1) == rangeBoundaries(1)) {
                markedRegion[k](1) = rangeBoundaries(0);
              }
              if (markedRegion[k](2) == rangeBoundaries(0)) {
                markedRegion[k](2) = rangeBoundaries(1);
              }
            }
          }

        }
        if (R.at<float>(Point(i, a)) == 0) {
          // mark the position as a member of the region
          R.at<float>(Point(i, a)) = 1;
          anyFalseInR++;
        }
      }
      if (noneInR == 0) {
        l++;
        slice << a,
                 rangeBoundaries(0),
                 rangeBoundaries(1);

        markedRegion.push_back(slice);
      }
    }

    iter++;

  } while(l<maxNumberKeyPoint && anyFalseInR < rows*cols); // until the number of landMark is reach or we explored every point

  //  Riordino il vettore delle markedRegion
  std::sort(markedRegion.begin(), markedRegion.end(), [](Eigen::Vector3f a, Eigen::Vector3f b) {
      return -a(0) > -b(0); // ordine crescente
  });

  float qLow, qUpper, qAngle;
  int potentialLandmark;
  float maxPower;
  float rangeOfLandmark;

  for (size_t q = 0; q < markedRegion.size(); q++) {
    qAngle = markedRegion[q](0);
    qLow = markedRegion[q](1);
    qUpper = markedRegion[q](2);

    for (int h = qLow; h < qUpper; h++) {
      if (qAngle > 0) {
        if(R.at<float>(Point(h, qAngle-1)) == 1 ){
          potentialLandmark = 1;
        }
      }
      if (qAngle < rows) {
        if(R.at<float>(Point(h, qAngle-1)) == 1 ){
          potentialLandmark = 1;
        }
      }
      if (potentialLandmark ==1) {
        maxPower = R.at<float>(Point(qLow, qAngle));
        rangeOfLandmark = qLow;
        for (int l = qLow+1; l < qUpper; l++) {
          if (R.at<float>(Point(l, qAngle))> maxPower) {
          rangeOfLandmark = l;
          }
        }
        keyPoint.at<float>(Point(rangeOfLandmark, qAngle)) = 1;
        break;
      }
    }
  }
  Mat cartKeyPoint;
  double maxRadius = 500.0;
  Point2f center( 500, 500);
  int flags =  WARP_INVERSE_MAP;

  warpPolar(keyPoint, cartKeyPoint, Size(1000,1000) , center, maxRadius,  flags);

  std::string superlandmark = "LANDMARK FUCK YEAH  ";
  cv::namedWindow(superlandmark, cv::WINDOW_AUTOSIZE);
  cv::imshow(superlandmark, cartKeyPoint); //show image.
  cv::waitKey();

  std::string land = "LANDMARK ";
  cv::namedWindow(land, cv::WINDOW_AUTOSIZE);
  cv::imshow(land, keyPoint); //show image.
  cv::waitKey();
  // for (size_t q = 0; q < markedRegion.size(); q++) {
  //       float angle = markedRegion[q](0);
  //       for (size_t qIter = 0; qIter < markedRegion.size(); qIter++) {
  //         if(angle - markedRegion[qIter](0) >=2 || markedRegion[qIter](0) - angle >=2 || markedRegion[qIter](0) - angle == 0 ){
  //           // Taking into consideration only adjacent angle --> angle-1 && angle+1
  //           continue;
  //         }
  //         if (angle - markedRegion[qIter](0) == 1 || markedRegion[qIter](0) - angle == 1) {
  //             // check if the region is not isolated
  //             low = markedRegion[qIter](1);
  //             upper = markedRegion[qIter](2);
  //         }
  //
  //
  //       }
  // }


  // int iterOnMarkedRegion;
  //
  // for (int angle=0; angle<cols; angle ++) {
  //   iterOnMarkedRegion=0;
  //   do {
  //     if(markedRegion[iterOnMarkedRegion](0) == angle){
  //
  //     }
  //   } while(/* condition */);
  // }
}
////////////////////////////////////////////////////////////////////////////////
Mat prewittOperator(Mat radarScanImage){

  Mat_<float> prewittX;
  Mat_<float> prewittY;
  Mat prewittQuadX, prewittQuadY, prewittQuadImage;
  Mat prewittImageProva;
  // Mat prewittImage;

  Mat_<float> kernelX(3,3);
  kernelX << 1, 0, -1, 1, 0,-1, 1, 0,-1;
  Mat_<float> kernelY(3,3);
  kernelY << 1, 1, 1, 0, 0, 0, -1, -1, -1;

  // Initialize arguments for the filter
  Point anchor = Point( -1, -1 ); // The position of the anchor relative to its kernel. The location Point(-1, -1) indicates the center by default.
  double delta = 0;
  int ddepth = -1; //The depth of dst. A negative value (such as −1) indicates that the depth is the same as the source
  int kernel_size;

  filter2D( radarScanImage,  prewittX, ddepth, kernelX, anchor, delta);
  filter2D( radarScanImage,  prewittY, ddepth, kernelY, anchor, delta);

  // Prewitt Filter
  // multiply(prewittX, prewittX, prewittQuadX);
  // multiply(prewittY, prewittY, prewittQuadY);
  // add(prewittQuadX, prewittQuadY, prewittQuadImage);

  // Mat prewittImage(prewittX.rows, prewittX.cols, CV_8UC1, Scalar(0));
  Mat_<float> prewittImage(prewittX.rows,prewittX.cols);

  for (int x = 0; x < prewittX.cols; x++) {
    for (int y = 0; y < prewittX.rows; y++) {
      float valuePixel, valuePixelX, valuePixelY, valuePixelSumAndSquare;

      valuePixelX = prewittX.at<float>(Point(x, y));
      valuePixelY = prewittY.at<float>(Point(x, y));
      valuePixel = sqrt(pow(valuePixelX,2) + pow(valuePixelY,2));
      prewittImage.at<float>(Point(x, y)) = valuePixel; // remember that for visualize the value in an image you have to scale valuePixel to 255.
    }
  }

  return prewittImage;
}

Mat getMatrixSPrime(Mat radarScanImage){

  float valuePixelSPrime, valuePixelS;
  Scalar meanValueS = cv::mean( radarScanImage );
  std::cout << "mean" <<  meanValueS(0)<< std::endl;
  Mat_<float> SPrime(radarScanImage.rows,radarScanImage.cols);

  for (int x = 0; x < radarScanImage.cols; x++) {
    for (int y = 0; y < radarScanImage.rows; y++) {

      valuePixelS = radarScanImage.at<float>(Point(x,y));
      valuePixelSPrime = valuePixelS -  meanValueS(0);
      SPrime.at<float>(Point(x,y)) = valuePixelSPrime;
    }
  }

  return SPrime;
}

Mat getMatrixH(Mat prewittImage, Mat SPrime){
  int rows = prewittImage.rows;
  int cols = prewittImage.cols;
  Mat_<float> HMatrix(rows,cols);

  float valuePixelPrewitt;
  float valuePixelSPrime;
  float valuePixelH;

  for (int x = 0; x < prewittImage.cols; x++) {
    for (int y = 0; y < prewittImage.rows; y++) {
      valuePixelPrewitt = prewittImage.at<float>(Point(x,y));
      valuePixelSPrime = SPrime.at<float>(Point(x,y));
      valuePixelH = (1 - valuePixelPrewitt)* valuePixelSPrime;
      HMatrix.at<float>(Point(x,y)) = valuePixelH;
    }
  }
  return HMatrix;
}

Vector3fVector getIndicesOfElementsInDescendingOrder(Mat prewittImage){

  int numberOfPowerReadings = (int) (prewittImage.cols * prewittImage.rows); // number of power readings in a Scan
  Vector3fVector indices(numberOfPowerReadings); // Initialize the dimension of indices vector
  float valuePixel;

  int i = 0;
  for (int x = 0; x < prewittImage.cols; x++) {
    for (int y = 0; y < prewittImage.rows; y++) {
      valuePixel = prewittImage.at<float>(Point(x,y));

      indices[i](0) = x;
      indices[i](1) = y;
      indices[i](2) = valuePixel;
      i++;
    }
  }
  // Ordino il vettore degli indici
  // sort using a lambda expression
  std::sort(indices.begin(), indices.end(), [](Eigen::Vector3f a, Eigen::Vector3f b) {
      return a(2) > b(2);
  });
  return indices;
}

Eigen::Vector2f findRangeBoundaries(float a, float r, Mat SPrime){

  float valuePixelSPrime;
  float rIterHigh = r+1;
  float rIterLow = r-1;
  float rHigh, rLow;
  int upperBound = 0;
  int lowBound = 0;
  //Get rHigh;
  valuePixelSPrime = SPrime.at<float>(Point(r, a));

  do {
    // valuePixelSPrime = SPrime.at<float>(Point(a, rIterHigh));
    valuePixelSPrime = SPrime.at<float>(Point(rIterHigh, a));
    if (valuePixelSPrime < 0) {
      rHigh = rIterHigh;
      upperBound = 1;
    }
    rIterHigh++;

    if (rIterHigh == SPrime.cols) { // Check if your out of border
      rHigh = rIterHigh;
      break;
    }

  } while(upperBound == 0);

  //Get rLow;
  do {
    valuePixelSPrime = SPrime.at<float>(Point(rIterLow, a));

    if (valuePixelSPrime < 0) {
      rLow = rIterLow;
      lowBound = 1;
    }

    if (rIterLow == 0) { // Check if your out of border
      rLow = rIterLow;
      break;
    }

    rIterLow--;
  } while(lowBound == 0);

  Eigen::Vector2f rangeBoundaries;
  rangeBoundaries(0) = rLow;
  rangeBoundaries(1) = rHigh;

  return rangeBoundaries;
}
