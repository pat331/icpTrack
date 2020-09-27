#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/highgui.h>

#include <numeric>

#include "../include/elaborationRadar.h"
#include "../include/defs.h"
using namespace std;
using namespace cv;

// Crop the image, removing the metadata
Mat cropRadarScan(Mat radarScanImage, int pointX, int pointY, int length, int height){
  // Setup a rectangle to define your region of interest
  // 11 is where start the data on powe reading.
  cv::Rect myROI(pointX, pointY, length, height);
  // Crop the full image to that image contained by the rectangle myROI
  // Note that this doesn't copy the data
  cv::Mat croppedRef(radarScanImage, myROI);
  cv::Mat cropped;
  // Copy the data into new matrix
  croppedRef.copyTo(cropped);
  return cropped;
}
//

//Filtering of the radarScan
Mat radarScanFilter(Mat radarScanImage, double thresh, double maxValue, double thresholdCanny1, double thresholdCanny2){

  cv::Mat dst, edges, detected_edges, radarScanImageCanny, radarScanImageFinal;

  // Canny edge detector;
  // blur( radarScanImage, detected_edges, Size(3,3) );
  // medianBlur ( radarScanImage, detected_edges, 7 );
  // bilateralFilter(radarScanImage, detected_edges, 15, 40, 40);
  // GaussianBlur(radarScanImage,detected_edges,Size(5,5),4,4);
  //
  // Canny(detected_edges, detected_edges, 20, 150, 3);
  // radarScanImageCanny = Scalar::all(0);
  // radarScanImage.copyTo( radarScanImageCanny, detected_edges);
  //
  // Threshold dopo il canning edge detector
  // blur( radarScanImageCanny, radarScanImageCanny, Size(5,1) ); // filtro migliore per ora
  // blur( radarScanImageCanny, radarScanImageCanny, Size(7,3) );
  // blur( radarScanImageCanny, radarScanImageCanny, Size(3,3) );
  threshold(radarScanImage, radarScanImageFinal, 70, maxValue, THRESH_BINARY);
  // threshold(radarScanImageCanny, radarScanImageFinal, 105, maxValue, THRESH_BINARY);
  //
  return radarScanImageFinal;
}

// Converto coordinate pixel in coordinate world
Eigen::Vector2f pixelToWorldCoord(Point pixelCoord, float pixelRange){
  Eigen::Vector2f worldCoord;
  float xP,yP;

  xP = pixelCoord.x - 1750;
  yP = pixelCoord.y - 200;

  worldCoord(0) = (xP * 164)/pixelRange;
  worldCoord(1) = (yP * 164)/pixelRange;

  return worldCoord;

}
//

// Vector2f heuristicLandmarks(Vector2f landMarkCartesian){
//   Vector2f heuristicLandmarks; // resistuiamo un vettore di posizioni dei nuovi landmark
//   int numberOfLandmarks = landMarkCartesian.size();
//   std::vector<int> checkedLandMarks(numberOfLandmarks, 0);
//   std::vector<int> landMarksInCluster(numberOfLandmarks, 0);
//   int numberOfLandmarksChecked = 0;
//
//   do {
//     std::vector<int> cluster;
//
//     // i prossimi due for trovano un cluster
//     for (int i=0; i<prevNumberOfLandmarks; i++){
//
//       if(landMarksInCluster[i] == 1){
//         continue; // controllo che il landMark non sia già all'interno di un cluster
//       }
//
//       if(checkedLandMarks[i] == 1){
//         continue;
//       }
//       numberOfLandmarksChecked++; // +1 sul numero di landMark controllati
//       checkedLandMarks[i] = 1; // marchiamo che il landmark i è stato giù osservato
//       cluster.push_back(i); // insieriamo nel cluster l'indice del landmark
//       Eigen::Vector2f posLandmark(landMarkCartesian.at(i)[0], landMarkCartesian.at(i)[1]);
//
//       for (int j = 0; j < numberOfLandmarks; j++) {
//
//         if(checkedLandMarks[j] == 1){
//           continue;
//         }
//         Eigen::Vector2f posLandmarkComparison(landMarkCartesian.at(j)[0], landMarkCartesian.at(j)[1]);
//         distFromLandMarks = (posLandmark - posLandmarkComparison).norm();
//         if (distFromLandMarks<3 /* da sostituire con threshold */) {
//           // aggiungo al cluster
//           //
//         }
//       }
//
//     }
//   } while(numberOfLandmarksChecked <= numberOfLandmarks);
//
//
// }
