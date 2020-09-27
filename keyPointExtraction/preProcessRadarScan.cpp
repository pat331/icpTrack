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

Mat prewittOperator(Mat radarScanImage){

  Mat prewittX, prewittY;
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
  Mat prewittImage(prewittX.rows, prewittX.cols, CV_8UC1, Scalar(0));

  for (int x = 0; x < prewittX.cols; x++) {
    for (int y = 0; y < prewittX.rows; y++) {
      double valuePixel, valuePixelX, valuePixelY, valuePixelSumAndSquare;

      valuePixelX = (int)prewittX.at<uchar>(Point(x, y));
      valuePixelY = (int)prewittY.at<uchar>(Point(x, y));
      valuePixel = sqrt(pow(valuePixelX,2) + pow(valuePixelY,2));
      prewittImage.at<uchar>(Point(x, y)) = (int)valuePixel;
    }
  }

  return prewittImage;
}

Mat getMatrixSPrime(Mat radarScanImage){

  int valuePixelSPrime, valuePixelS;
  Scalar meanValueS = cv::mean( radarScanImage );
  Mat SPrime(radarScanImage.rows, radarScanImage.cols, CV_8UC1, Scalar(0));

  for (int x = 0; x < radarScanImage.cols; x++) {
    for (int y = 0; y < radarScanImage.rows; y++) {

      valuePixelS = (int)radarScanImage.at<uchar>(Point(x,y));
      valuePixelSPrime = valuePixelS - (int) meanValueS(0);
      SPrime.at<uchar>(Point(x,y)) = valuePixelSPrime;

    }
  }

  return SPrime;
}

// Mat getMatrixH(Mat prewittImage, Mat SPrime){
//
//
// }

Vector2fVector getIndicesOfElementsInDescendingOrder(Mat prewittImage){

  for (int x = 0; x < prewittImage.cols; x++) {
    for (int y = 0; y < prewittImage.rows; y++) {
      /* code */
    }
  }

}
