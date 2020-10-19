#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>

#include "singleValueDecomposition2D.h"
#include "defs.h"


using namespace std;
using namespace cv;
using namespace pr;

void singularValueDecomposition2D(Eigen::Matrix<float, 2, 2> R){

  Eigen::Matrix<float, 2, 2> U;
  Eigen::Matrix<float, 2, 2> Sigma;
  Eigen::Matrix<float, 2, 2> V;

  U = computationOfU(R(0,0), R(0,1), R(1,0), R(1,1));
  Sigma = computationOfSigma(R(0,0), R(0,1), R(1,0), R(1,1));
  V = computationOfV(R(0,0), R(0,1), R(1,0), R(1,1));



}
Eigen::Matrix<float, 2, 2> computationOfU(float a, float b, float c, float d){

  float theta = 0.5*atan2(2*a*c + 2*b*d, pow(a,2) + pow(b,2) - pow(c,2) - pow(d,2));
  Eigen::Matrix<float, 2, 2> U;
  U << cos(theta), sin(theta), -sin(theta), cos(theta);

  return U;
}
Eigen::Matrix<float, 2, 2> computationOfSigma(float a, float b, float c, float d){

  float S1 = pow(a,2) + pow(b,2) + pow(c,2) + pow(d,2);
  float S2 = sqrt(pow(pow(a,2) + pow(b,2) - pow(c,2) - pow(d,2),2) + 4*pow(a*c + b*d,2));

  float sigma1 = sqrt((S1+S2)*0.5);
  float sigma2 = sqrt((S1-S2)*0.5);

  Eigen::Matrix<float, 2, 2> Sigma;
  Sigma << sigma1, 0, 0, sigma2;

  return Sigma;
}
Eigen::Matrix<float, 2, 2> computationOfV(float a, float b, float c, float d){

  float phi = 0.5*atan2(2*a*b + 2*c*d, pow(a,2) - pow(b,2) + pow(c,2) - pow(d,2));
  float theta = 0.5*atan2(2*a*c + 2*b*d, pow(a,2) + pow(b,2) - pow(c,2) - pow(d,2));

  float s11 = (a*cos(theta) + c*sin(theta))*cos(phi) + (b*cos(theta) + d*sin(theta))*sin(phi);
  float s22 = (a*sin(theta) - c*cos(theta))*sin(phi) + (-b*sin(theta) + d*cos(theta))*cos(phi);

  Eigen::Matrix<float, 2, 2> V;
  V << (s11/abs(s11))*cos(phi), (-s22/abs(s22))*sin(phi), (s11/abs(s11))*sin(phi), (s22/abs(s22))*cos(phi);
  return V;
}
