#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

// My function includes
#include "../include/loadRadar.h"
#include "../include/elaborationRadar.h"
// #include "elaborationRadar.h"
#include "../include/points_utils.h"
#include "../include/defs.h"
#include "../include/data_association.h"
#include "../include/smicp_solver.h"

#include "preProcessRadarScan.h"
#include "dataAssociationRadar.h"
#include "rigidBodyMotion.h"
#include "singleValueDecomposition2D.h"
#include "dataAssociationSURF.h"
#include "localMap.h"
#include "mapping.h"

#define refinement

const int img_width = 1000;
const int img_height = 1000;

int local_scale = 1;
int local_height_offset = img_height/1.5;

// const float gating_thres = 0.2;
const float gating_thres = 1;
const float lbf_thres = 0.001;

using namespace pr;

int main(int argc, char *argv[]){

  //Declaration
  float pixelRange;
  Point pnt;
  Eigen::Vector2f pntWorld;
  //


  string gtCSVFolder = "/home/luca/Documents/tesi/validation/2019-01-15-12-52-32-radar-oxford-10k-partial_Navtech_CTS350-X_Radar_Optimised_SE2_Odometry/2019-01-15-12-52-32-radar-oxford-10k-partial/gt/radar_odometry.csv";
  // string gtCSVFolder = "/home/luca/Documents/tesi/validation/2019-01-15-12-52-32-radar-oxford-10k-partial_Navtech_CTS350-X_Radar_Optimised_SE2_Odometry/2019-01-15-12-52-32-radar-oxford-10k-partial/gt/onlyCoord_radar_odometry.csv";
  // vector<vector<double>> data = parse2DCsvFile(gtCSVFolder);
  // Read gt csv
  std::vector<std::pair<std::string, std::vector<double>>> gt = read_csv(gtCSVFolder);
  // std::cerr << "gt "<< gt[0].second.size() << '\n';
  Vector2fVector gtPose;
  Eigen::Vector2f initialPose;
  initialPose << 0, 0;
  gtPose.push_back(initialPose);

  std::vector<double> dTheta;
  for (size_t i = 0; i < gt[0].second.size(); i++) {
    dTheta.push_back(gt[7].second[i]);
  }
  std::vector<double> dx;
  for (size_t i = 0; i < gt[0].second.size(); i++) {
    dx.push_back(gt[2].second[i]);
  }
  std::vector<double> dy;
  for (size_t i = 0; i < gt[0].second.size(); i++) {
    dy.push_back(gt[3].second[i]);
  }

  Eigen::Matrix<float, 2, 2> Rf;
  Eigen::Matrix<float, 2, 2> RPrec;
  Eigen::Matrix<float, 2, 2> RSucc;
  RPrec << 1,0,0,1;
  Eigen::Vector2f tf;
  Eigen::Vector2f tPrec;
  Eigen::Vector2f tSucc;
  tPrec << 0,0;

  for (size_t i = 0; i <1935; i++) {
    Eigen::Vector2f pos;
    RSucc << cos(dTheta[i]),-sin(dTheta[i]),sin(dTheta[i]),cos(dTheta[i]);
    tSucc << dx[i]*0.3,dy[i]*0.3;
    pos = RSucc*RPrec*initialPose + RPrec*tSucc + tPrec;
    tPrec = RPrec*tSucc + tPrec;
    RPrec = RPrec*RSucc;
    pos(0) = pos(0)+500;
    pos(1) = pos(1)+500;
    // if (i>=100) {
    //   gtPose.push_back(pos);
    // }
    gtPose.push_back(pos);
  }

  RGBImage ground_truth(1000, 1000);
  ground_truth.create(1000, 1000);
  ground_truth=cv::Vec3b(255,255,255);
  // drawPoints(local_image, provaDisp, cv::Scalar(255,0,0),1);
  drawPoints(ground_truth, gtPose, cv::Scalar(0,0,255),1);
  cv::imshow("Scan matcher", ground_truth);
  waitKey();

  // Retrieve all the png file of the radar scan in the folder //
  // string radaScanPngFilesFolder = "/home/luca/Documents/tesi/oxford_radar_robotcar_dataset_sample_small/2019-01-10-14-36-48-radar-oxford-10k-partial/radar/*.png";
  // string radaScanPngFilesFolder = "/home/luca/Documents/tesi/2019-01-10-11-46-21-radar-oxford-10k_Navtech_CTS350-X_Radar/2019-01-10-11-46-21-radar-oxford-10k/radarPartial/*.png";
  // string radaScanPngFilesFolder = "/home/luca/Documents/tesi/2019-01-10-11-46-21-radar-oxford-10k_Navtech_CTS350-X_Radar/2019-01-10-11-46-21-radar-oxford-10k/radarProva/*.png";
  // string radaScanPngFilesFolder = "/home/luca/Documents/tesi/2019-01-10-11-46-21-radar-oxford-10k_Navtech_CTS350-X_Radar/2019-01-10-11-46-21-radar-oxford-10k/radar/*.png";
  string radaScanPngFilesFolder = "/home/luca/Documents/tesi/validation/2019-01-15-12-52-32-radar-oxford-10k-partial/radar/*.png";
  vector<String> pathRadarScanPngFiles;
  pathRadarScanPngFiles = loadPathRadarScanPngFiles(radaScanPngFilesFolder);
  //
  // How many radar scan we have
  int numberOfRadarScan = 0;
  numberOfRadarScan = pathRadarScanPngFiles.size();
  //
  // Da ogni radarscan filtrando l'immagine prendo i punti che trovo e li metto tutti dentro un vettore
  std::vector<Vector2fVector> landMarkCartesian;
  string dataFilePng;
  string dataFilePngSucc, dataFilePngSucc2;
  string dataBikeStrike, dataBikeStrikePrewitt;

  cv::Mat key1, key2, key3;

  Eigen::Matrix<float, 2, 2> RotMap;
  RotMap << 1,0,0,1;
  Eigen::Vector2f tMap;
  tMap << 0,0;
  SE2 mappingMotion;
  mappingMotion.R = RotMap;
  mappingMotion.t = tMap;

  SE2 partialMotion;
  partialMotion.R = RotMap;
  partialMotion.t = tMap;

  Eigen::Matrix<float, 2, 2> Rtot;
  Rtot << 1,0,0,1;
  Eigen::Matrix<float, 2, 2> R1;
  Rtot << 1,0,0,1;
  Eigen::Matrix<float, 2, 2> R2;
  Rtot << 1,0,0,1;
  Eigen::Matrix<float, 2, 2> RFrameToWorld;
  Rtot << 1,0,0,1;
  Eigen::Matrix<float, 2, 2> RFrameToFrame;
  Rtot << 1,0,0,1;
  Eigen::Vector2f translationVectorTot;
  translationVectorTot << 0,0;

  Eigen::Vector2f translationVectorFrameToFrame;
  translationVectorTot << 0,0;
  Eigen::Vector2f translationVectorFrameToWorld;
  translationVectorTot << 0,0;

  Eigen::Vector2f t1;
  translationVectorTot << 0,0;
  Eigen::Vector2f t2;
  translationVectorTot << 0,0;
  // Itero su tutti i radarscan della cartella
  SE2 motion;
  LocalMap map, map1;
  Mapping mapping;
  int firstScan = 1600;

  for (int i = firstScan; i < 2100; i++) {
    dataFilePng = pathRadarScanPngFiles[i];
    dataFilePngSucc = pathRadarScanPngFiles[i+1];
    dataFilePngSucc2 = pathRadarScanPngFiles[i+2];

    //-- Step 1: Detect the keypoints using SURF Detector
    cv::Mat provaImageSurf = imread(dataFilePng, cv::IMREAD_GRAYSCALE);
    cv::Mat provaImageSurf2 = imread(dataFilePngSucc, cv::IMREAD_GRAYSCALE);
    cv::Mat provaImageSurf3 = imread(dataFilePngSucc2, cv::IMREAD_GRAYSCALE);

    Mat prova, prova2, prova3;
    Mat  provaBlur, provaBlur2, cart, cartSucc, cartSucc2, cartBlur, cartBlurSucc, cartBlurSucc2;
    // Mat provaBlura,provaBlurb,provaBlurc,provaBlurd;

    prova =cropRadarScan(provaImageSurf, 11, 0, 3000, 400);
    prova2 =cropRadarScan(provaImageSurf2, 11, 0, 3000, 400);
    prova3 =cropRadarScan(provaImageSurf3, 11, 0, 3000, 400);

    double maxRadius = 400.0;
    Point2f center( 400, 400);
    int flags = INTER_LINEAR + WARP_FILL_OUTLIERS + WARP_INVERSE_MAP;

    warpPolar(prova, cart, Size(800,800) , center, maxRadius,  flags);
    warpPolar(prova2, cartSucc, Size(800,800) , center, maxRadius,  flags);
    warpPolar(prova3, cartSucc2, Size(800,800) , center, maxRadius,  flags);

    threshold( cart, cart, 35, 255, 3 );
    threshold( cartSucc, cartSucc, 35, 255, 3 );
    threshold( cartSucc2, cartSucc2, 35, 255, 3 );

    blur( cart, cartBlur, Size(3,3) );
    blur( cartSucc, cartBlurSucc, Size(3,3) );
    blur( cartSucc2, cartBlurSucc2, Size(3,3) );

    if (i==firstScan) {
      KeyAndDesc kd1, kd2;
      kd1 = mapping.findScanKeyPoint(cartBlur);
      mapping.initMap(kd1, mappingMotion);
      kd2 = mapping.findScanKeyPoint(cartBlurSucc);
      std::vector<DMatch> match;
      match = mapping.matchMap(kd2);
      mappingMotion = mapping.scanMap(kd2,match);
      mapping.robotMotion();
      mapping.insertKeyFrame(kd2);
      continue;
    }

    KeyAndDesc kd1,kd2;
    kd1 = mapping.findScanKeyPoint(cartBlur);
    kd2 = mapping.findScanKeyPoint(cartBlurSucc);
    std::vector<DMatch> match;
    match = mapping.matchMap(kd2);
    mappingMotion = mapping.scanMap(kd2,match);
    mapping.robotMotion();

    Eigen::Matrix<float, 2, 2> Rotation_gt;
    Eigen::Vector2f translation_gt;
    Rotation_gt << cos(dTheta[i+1]),-sin(dTheta[i+1]),sin(dTheta[i+1]),cos(dTheta[i+1]);
    translation_gt << dx[i+1],dy[i+1];
    mapping.errorEstimation(Rotation_gt, translation_gt);

    bool reset;
    reset = mapping.resetMap();
    if (reset == 0) {
      mapping.insertKeyFrame(kd2);
      // mapping.clearMap();
      // SE2 restartMotion;
      // restartMotion.R << 1,0,0,1;
      // restartMotion.t << 0,0;
      // mapping.initMap(kd2, restartMotion);
    } else if (reset == 1) {
      // mapping.clearMap();
      // SE2 restartMotion;
      // restartMotion.R << 1,0,0,1;
      // restartMotion.t << 0,0;
      // mapping.initMap(kd2, restartMotion);
      mapping.clearMap();
      SE2 restartMotion;
      restartMotion.R = mappingMotion.R;
      restartMotion.t = mappingMotion.t;
      mapping.initMap(kd1, restartMotion);
      mapping.insertFrameReset(kd2, mappingMotion);
    }
    std::cerr << "i = "<< i << '\n';
  }
  mapping.dispMotion();
  mapping.dispError();

  return 0;

  for(int i = firstScan; i < 1; i++){
    dataFilePng = pathRadarScanPngFiles[i];
    dataFilePngSucc = pathRadarScanPngFiles[i+1];
    dataFilePngSucc2 = pathRadarScanPngFiles[i+2];
    dataBikeStrike = "/home/luca/Downloads/Bikesgray.jpg";
    dataBikeStrikePrewitt = "/home/luca/Downloads/Bikesgray_prewitt.JPG";

    cv::Mat cropped, croppedSucc, croppedSucc2;
    cv::Mat cart, cartSucc, cartSucc2;
    cv::Mat cartFiltered;
    cv::Mat locations;   // output, locations of non-zero pixels


    //-- Step 1: Detect the keypoints using SURF Detector
    cv::Mat provaImageSurf = imread(dataFilePng, cv::IMREAD_GRAYSCALE);
    cv::Mat provaImageSurf2 = imread(dataFilePngSucc, cv::IMREAD_GRAYSCALE);
    cv::Mat provaImageSurf3 = imread(dataFilePngSucc2, cv::IMREAD_GRAYSCALE);

    Mat prova;
    Mat prova2;
    Mat prova3;
    Mat  provaBlur, provaBlur2, cartBlur, cartBlurSucc, cartBlurSucc2;
    // Mat provaBlura,provaBlurb,provaBlurc,provaBlurd;

    prova =cropRadarScan(provaImageSurf, 11, 0, 3000, 400);
    prova2 =cropRadarScan(provaImageSurf2, 11, 0, 3000, 400);
    prova3 =cropRadarScan(provaImageSurf3, 11, 0, 3000, 400);

    double maxRadius = 400.0;
    Point2f center( 400, 400);
    int flags = INTER_LINEAR + WARP_FILL_OUTLIERS + WARP_INVERSE_MAP;

    warpPolar(prova, cart, Size(800,800) , center, maxRadius,  flags);
    warpPolar(prova2, cartSucc, Size(800,800) , center, maxRadius,  flags);
    warpPolar(prova3, cartSucc2, Size(800,800) , center, maxRadius,  flags);

    threshold( cart, cart, 35, 255, 3 );
    threshold( cartSucc, cartSucc, 35, 255, 3 );
    threshold( cartSucc2, cartSucc2, 35, 255, 3 );

    blur( cart, cartBlur, Size(3,3) );
    blur( cartSucc, cartBlurSucc, Size(3,3) );
    blur( cartSucc2, cartBlurSucc2, Size(3,3) );

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 20;
    Ptr<SURF> detector = SURF::create( minHessian);
    std::vector<KeyPoint> keypoints1, keypoints2, keypoints3;
    Mat descriptors1, descriptors2, descriptors3;

    detector->detectAndCompute( cartBlur, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( cartBlurSucc, noArray(), keypoints2, descriptors2 );
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    // if (i==firstScan) {
      // Mat mapPoints1;
      // drawKeypoints( cartBlur, keypoints1, mapPoints1 );
      // //-- Show detected (drawn) keypoints
      // imshow("First map 1", mapPoints1 );
      // waitKey();
    // }


    if (i==firstScan) {
      map.initFirstMap(keypoints1,descriptors1);
      map.fillNKS(keypoints1.size());
      // map1.initFirstMap(keypoints1,descriptors1);
      R1.setIdentity();
      t1 << 0,0;
    }
    // map.initFirstMap(keypoints1,descriptors1);

    // -- Draw matches
    // Mat img_matches33;
    // // drawMatches( cart, keypoints1, cartSucc, keypoints2, good_matches, img_matches, Scalar::all(-1),
    // //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // drawMatches( cartBlur, keypoints1, cartBlurSucc, keypoints2, good_matches, img_matches33, Scalar::all(-1),
    //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // imshow("Matches", img_matches33 );
    // waitKey();

    std::vector<int> maxClique;
    maxClique = createConsistencyMatrix(keypoints1, keypoints2, good_matches);
    // maxClique = createConsistencyMatrix(keypoints_object, keypoints_scene, good_matchesORB);

    std::vector<DMatch> ultimate_matches;
    for (size_t i = 0; i < good_matches.size(); i++) {
      if (maxClique[i] == 1) {
        ultimate_matches.push_back(good_matches[i]);
      }
    }

    //
    // Mat img_matches;
    // // drawMatches( cart, keypoints1, cartSucc, keypoints2, good_matches, img_matches, Scalar::all(-1),
    // //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // drawMatches( cartBlur, keypoints1, cartBlurSucc, keypoints2, ultimate_matches, img_matches, Scalar::all(-1),
    //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // imshow("Matches", img_matches );
    // waitKey();
    std::cerr << "i = "<< i << '\n';
    std::cerr << "ultimate matche size  "<< ultimate_matches.size() << '\n';
    if (ultimate_matches.size() <3) {
      std::cerr << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << '\n';
      std::cerr << "Ripeto scan match con scan successivo " << '\n';

      Ptr<SURF> detector2 = SURF::create( minHessian);
      detector2->detectAndCompute( cartBlurSucc2, noArray(), keypoints3, descriptors3 );

      Ptr<DescriptorMatcher> matcher2 = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
      std::vector< std::vector<DMatch> > knn_matches;
      matcher2->knnMatch( descriptors1, descriptors3, knn_matches, 2 );
      //-- Filter matches using the Lowe's ratio test
      const float ratio_thresh = 0.7f;
      std::vector<DMatch> good_matches;
      for (size_t i = 0; i < knn_matches.size(); i++)
      {
          if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
          {
              good_matches.push_back(knn_matches[i][0]);
          }
      }

      std::vector<int> maxClique2;
      maxClique2 = createConsistencyMatrix(keypoints1, keypoints3, good_matches);
      ultimate_matches.clear();
      for (size_t i = 0; i < good_matches.size(); i++) {
        if (maxClique2[i] == 1) {
          ultimate_matches.push_back(good_matches[i]);
        }
      }
      if (ultimate_matches.size()<3) {
        std::cerr << "###############################" << '\n';
        std::cerr << "Problema non risolto" << '\n';
        std::cerr << "ultimate_matches seconda prova "<< ultimate_matches.size() << '\n';
      }else{
        std::cerr << "ultimate_matches seconda prova "<< ultimate_matches.size() << '\n';
      }
      keypoints2.clear();
      keypoints2=keypoints3;
    }

    // PARTE FUNZIONANTE PER LA CREAZIONE DELLA MAPPA
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::cerr << "++++++++++++++++++++++++++++++++++++" << '\n';
    Eigen::Matrix<float, 2, 2> Rf;

    Rf = rigidBodyMotionSurf(keypoints1, keypoints2, ultimate_matches); // questa è la rotazione che porta da ref1->ref2
    Eigen::Vector2f mean1;
    mean1 = meanScanSurf1(keypoints1, ultimate_matches);
    Eigen::Vector2f mean2;
    mean2 = meanScanSurf2(keypoints2, ultimate_matches);


    Eigen::Vector2f translationVectorf;
    translationVectorf = mean2 - Rf * mean1;
    // Error estimation
    Eigen::Matrix<float, 2, 2> Rotation_gt;
    Eigen::Vector2f translation_gt;
    Rotation_gt << cos(dTheta[i]),-sin(dTheta[i]),sin(dTheta[i]),cos(dTheta[i]);
    translation_gt << dx[i],dy[i];
    map.errorEstimation(Rotation_gt,translation_gt,Rf,translationVectorf);
    //
    R2 = Rf.inverse();
    t2 = -R2*translationVectorf;
    Rtot = R1*R2;
    translationVectorTot = R1*t2+t1;
    R1 = Rtot;
    t1 = translationVectorTot;

    SE2 robMotion;
    robMotion.R = Rtot;
    robMotion.t = translationVectorTot;

    map.robotMotion(robMotion);

    std::vector<KeyPoint> provaKeyMap;

    // if (ultimate_matches.size()<35) {
    //   map.insertKeyFrame(keypoints2,descriptors2,Rtot,translationVectorTot);
    //   map.fillNKS(keypoints2.size());
    // }
    // map.insertKeyFrame(keypoints2,descriptors2,Rtot,translationVectorTot);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // std::cerr << "################################" << '\n';
    // if (i==firstScan) {
    //   motion.R.setIdentity();
    //   motion.t << 0,0;
    // }
    //
    //
    // std::vector<DMatch> matchWithMap;
    // matchWithMap = map1.matchingWithMap(keypoints2,descriptors2);
    // std::vector<int> indexLandmarkAssociated(keypoints2.size(),0);
    // for (size_t i = 0; i < matchWithMap.size(); i++) {
    //   indexLandmarkAssociated[matchWithMap[i].trainIdx] = 1;
    // }
    // std::cerr << "/* matchingWithMap size  */ "<< matchWithMap.size() << '\n';
    // float check =0;
    // for (size_t i = 0; i < indexLandmarkAssociated.size(); i++) {
    //   check+=indexLandmarkAssociated[i];
    // }
    //
    // SE2 motion2;
    // motion2 = map1.trackLocalMap(keypoints1,descriptors1,keypoints2,descriptors2,motion,matchWithMap,indexLandmarkAssociated);
    // if (motion2.t(0) < -999 && motion2.t(1) < -999) { // failure in map-scan matching
    //   Eigen::Vector2f translationVectorf;
    //   translationVectorf = mean2 - Rf * mean1;
    //   R2 = Rf.inverse();
    //   t2 = -R2*translationVectorf;
    //   motion.R = motion.R*R2;
    //   motion.t = motion.R*t2 + motion.t;
    //   map1.insertKeyFrame2(keypoints2,descriptors2,motion.R,motion.t);
    //   map1.robotMotion(motion);
    //   // motion.R = motion2;
    // }else{
    //   map1.robotMotion(motion2);
    //   motion.R = motion2.R;
    //   motion.t = motion2.t;
    // }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // map.trackLocalMap(keypoints2,descriptors2,ultimate_matches,Rtot,translationVectorTot);

    // //-- Draw final matches
    // Mat img_matches2;
    // // drawMatches( cartBlur, keypoints1, cartBlurSucc, keypoints2, ultimate_matches, img_matches2, Scalar::all(-1),
    // //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // drawMatches( cartBlur, keypoints1, cartBlurSucc, keypoints2, ultimate_matches, img_matches2, Scalar::all(-1),
    //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // imshow("Matches ultimate ", img_matches2 );
    // waitKey();
    //
    // /////////////////////////////////////////////////////////////////////////////
    // // Track third scan for localMap try
    // Ptr<SURF> detector2 = SURF::create( minHessian );
    // detector2->detectAndCompute( cartBlurSucc2, noArray(), keypoints3, descriptors3 );
    // Ptr<DescriptorMatcher> matcher2 = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    // std::vector< std::vector<DMatch> > knn_matches2;
    // matcher2->knnMatch( descriptors2, descriptors3, knn_matches2, 2 );
    // std::vector<DMatch> good_matches2;
    // for (size_t i = 0; i < knn_matches2.size(); i++)
    // {
    //     if (knn_matches2[i][0].distance < ratio_thresh * knn_matches2[i][1].distance)
    //     {
    //         good_matches2.push_back(knn_matches2[i][0]);
    //     }
    // }
    // std::vector<int> maxClique2;
    // maxClique2 = createConsistencyMatrix(keypoints2, keypoints3, good_matches2);
    //
    // std::vector<DMatch> ultimate_matches2;
    // for (size_t i = 0; i < good_matches2.size(); i++) {
    //   if (maxClique2[i] == 1) {
    //     ultimate_matches2.push_back(good_matches2[i]);
    //   }
    // }
    //-- Draw matches
    // Mat img_matches;
    // // drawMatches( cart, keypoints1, cartSucc, keypoints2, good_matches, img_matches, Scalar::all(-1),
    // //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // drawMatches( cartBlurSucc, keypoints2, cartBlurSucc2, keypoints3, ultimate_matches2, img_matches, Scalar::all(-1),
    //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // imshow("Matches ultimate_matches", img_matches );
    // waitKey();

    // std::cerr << "ultimate_matches2 main "<< ultimate_matches2.size() << '\n';
    // Rf = rigidBodyMotionSurf(keypoints2, keypoints3, ultimate_matches2);
    // // std::cerr << "Rf "<< Rf << '\n';
    // Rtot = Rtot*Rf;
    // mean1 = meanScanSurf1(keypoints2, ultimate_matches2);
    // mean2 = meanScanSurf2(keypoints3, ultimate_matches2);
    // translationVectorf = mean2 - Rf * mean1;
    // translationVectorTot += translationVectorf;
    // std::cerr << "translation vector 2 "<< translationVectorTot << '\n';
    // std::cerr << "R tot main "<< Rtot << '\n';

    // map.trackLocalMap(keypoints3,descriptors3, Rtot, translationVectorTot);
    // map.dispMap();
    /////////////////////////////////////////////////////////////////////////////

    // RGBImage local_image(img_width, img_height);
    // local_image.create(img_width, img_height);
    // local_image=cv::Vec3b(255,255,255);
    //
    // Vector2fVector scan_for_disp;
    // for (int i=0; i<ultimate_matches.size(); i++){
    //   Eigen::Vector2f cartesian_point(keypoints1[ultimate_matches[i].queryIdx].pt.x, keypoints1[ultimate_matches[i].queryIdx].pt.y);
    //   scan_for_disp.push_back(cartesian_point);
    // }
    // drawPoints(local_image, scan_for_disp, cv::Scalar(255,0,0),1);
    //
    // Vector2fVector post_scan_for_disp;
    // for (int i=0; i<ultimate_matches.size(); i++){
    //   Eigen::Vector2f cartesian_point(keypoints2[ultimate_matches[i].trainIdx].pt.x, keypoints2[ultimate_matches[i].trainIdx].pt.y);
    //   post_scan_for_disp.push_back(cartesian_point);
    // }
    // drawPoints(local_image, post_scan_for_disp, cv::Scalar(0,255,0),1);
    //
    // Vector2fVector transformed_scan_for_disp;
    // Eigen::Vector2f positionLand;
    // for (int i=0; i<ultimate_matches.size(); i++){
    //   // Eigen::Vector2f transformed_point = init_transform.inverse()*scans_cartesian.at(iter).at(i);
    //
    //   positionLand(0) = keypoints1[ultimate_matches[i].queryIdx].pt.x;
    //   positionLand(1) = keypoints1[ultimate_matches[i].queryIdx].pt.y;
    //   Eigen::Vector2f transformed_point = Rf * positionLand + translationVectorf;
    //
    //   Eigen::Vector2f cartesian_point(transformed_point(0), transformed_point(1));
    //   transformed_scan_for_disp.push_back(cartesian_point);
    //
    // }
    // // 0,150,255
    // // drawPoints(local_image, transformed_scan_for_disp, cv::Scalar(0,0,255),1);
    //
    // cv::imshow("Scan matcher", local_image);
    //
    // waitKey(0);

  }
  map.dispMotion();
  // map.dispMap();
  // map1.dispMap();
  return 0;


  // Initialize images
  // Local view
  RGBImage local_image(img_width, img_height);
  local_image.create(img_width, img_height);
  local_image=cv::Vec3b(255,255,255);
  // Global view
  RGBImage map_image(img_width, img_height);
  map_image.create(img_width, img_height);
  map_image=cv::Vec3b(255,255,255);

  RGBImage refined_map_image(img_width, img_height);
  refined_map_image.create(img_width, img_height);
  refined_map_image=cv::Vec3b(255,255,255);

  SMICPSolver solver;

  // Poses from first estimate
  std::vector<Eigen::Isometry2f, Eigen::aligned_allocator<Eigen::Isometry2f>> poses;
  Eigen::Isometry2f curr_pose;
  curr_pose.setIdentity();
  poses.push_back(curr_pose);

  // Poses from refinement
  std::vector<Eigen::Isometry2f, Eigen::aligned_allocator<Eigen::Isometry2f>> refined_poses;
  Eigen::Isometry2f refined_curr_pose;
  refined_curr_pose.setIdentity();
  refined_poses.push_back(refined_curr_pose);

  // Poses for drawing (scaled and translated to image center)
  // local view
  Vector2fVector pose_vec;
  // global view
  Vector2fVector pose_vec_map;
  // insert 0 pose into vec
  Eigen::Vector2f point;
  point << t2vEuler2D(curr_pose)(0)*local_scale+img_width/2,
           t2vEuler2D(curr_pose)(1)*local_scale+img_height/2;
  pose_vec.push_back(point);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Iterazione su tutti i radar scan ( presi due alla volta(?))
  int step = 1;
  // for (int iter = 1385; iter < num_poses; iter+=step)
  for (int iter = step; iter < 1; iter+=step)
  // int iter = 200;
  {
    // clear local view
    local_image=cv::Vec3b(255,255,255);

    //------------------------------------------
    // Show 2 scans
    // Scale and translate scans for better display
    Vector2fVector scan_for_disp;
    int numberOfLandmarks = landMarkCartesian.at(iter).size(); // numero di landmark nel radarscan "iter"
    std::cout << "landmarks nello scan di partenza "<< numberOfLandmarks  << std::endl;
    for (int i=0; i<numberOfLandmarks; i++){
      Eigen::Vector2f cartesian_point(landMarkCartesian.at(iter).at(i)[0]*local_scale + img_width/2, landMarkCartesian.at(iter).at(i)[1]*local_scale + img_height/2);
      scan_for_disp.push_back(cartesian_point);
    }
    drawPoints(local_image, scan_for_disp, cv::Scalar(0,150,255),1);

    Vector2fVector prev_scan_for_disp;
    int prevNumberOfLandmarks = landMarkCartesian.at(iter-step).size();
    std::cout << "landmarks nello scan di di arrivo "<< prevNumberOfLandmarks  << std::endl;
    for (int i=0; i<prevNumberOfLandmarks; i++){
      Eigen::Vector2f cartesian_point(landMarkCartesian.at(iter-step).at(i)[0]*local_scale + img_width/2, landMarkCartesian.at(iter-step).at(i)[1]*local_scale + img_height/2);
      prev_scan_for_disp.push_back(cartesian_point);
    }
    // drawPoints(local_image, prev_scan_for_disp, cv::Scalar(255,0,0),1);
    //----------------------------------------
    // Tentativo di trovare dei landmark nuovi dopo quelli del threshold per riuscire a fare un data association efficace
    // tentativo di usare il sift

    // Vector2f prevHeuristicLandmarks;
    // prevHeuristicLandmarks = heuristicLandmarks(landMarkCartesian.at(iter-step));
    //
    // Compute rough association (with tons of false matches)
    IntPairVector rough_correspondences;
    computeCorrespondences(rough_correspondences, landMarkCartesian.at(iter-step), landMarkCartesian.at(iter), gating_thres, lbf_thres);
    std::cout << "Quante corrispondenze abbiamo trovato "<< rough_correspondences.size() << std::endl;
    // #ifdef refinement
    //   drawCorrespondences(local_image, prev_scan_for_disp, scan_for_disp, rough_correspondences, cv::Scalar(0,255,0));
    // #endif

    // Matching
    // std::cout << "init transform \n" << init_transform.linear() << "\n" << init_transform.translation() << std::endl;
    solver.init(Eigen::Isometry2f::Identity(),landMarkCartesian.at(iter-step),landMarkCartesian.at(iter));
    for (int n_round=0; n_round < 10; n_round++){
      solver.oneRound(rough_correspondences, false);
    }
    curr_pose = solver.transform()*poses.back();
    poses.push_back(curr_pose);
    std::cout << "First estimate \n";
    std::cout << "step dx: \n" << t2v(solver.transform()) << std::endl;
    std::cout << "tot dx: \n" << t2v(curr_pose) << std::endl;

    refined_curr_pose = solver.transform()*refined_poses.back();
    refined_poses.push_back(refined_curr_pose);

    // Draw result in local view
    Vector2fVector transformed_scan_for_disp;
    for (int i=0; i<numberOfLandmarks; i++){
      Eigen::Vector2f transformed_point = solver.transform().inverse()*landMarkCartesian.at(iter).at(i);
      // Eigen::Vector2f transformed_point = init_transform.inverse()*scans_cartesian.at(iter).at(i);
      Eigen::Vector2f cartesian_point(transformed_point[0]*local_scale + img_width/2, transformed_point[1]*local_scale + img_height/2);
      transformed_scan_for_disp.push_back(cartesian_point);
    }
    // 0,150,255
    // drawPoints(local_image, transformed_scan_for_disp, cv::Scalar(255,0,255),1);
    // drawPoints(local_image, transformed_scan_for_disp, cv::Scalar(0,150,255),1);
    cv::imshow("Scan matcher", local_image);
    cv::waitKey(0);
  }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//   std::vector<Vector2fVector> landMarkCartesian;
// for (int i=0; i<num_poses; i++)
//   {
//     landMarkCartesian.push_back(Vector2fVector());
//     for (int j=0; j<num_angles; j++)
//       {
//         Eigen::Vector2f scan_cartesian;
//         scan_cartesian(0) = scans(i,j)*cos(min_angle + j*angle_increment);
//         scan_cartesian(1) = scans(i,j)*sin(min_angle + j*angle_increment);
//         landMarkCartesian.back().push_back(scan_cartesian);
//       }
//     }

  //////////////////////////////////////////////////////////////////////////////
  // cv::Mat dst;
  // string dataFilePng = pathRadarScanPngFiles[1];
  // // // cv::Mat RadarScanImage = imread(dataFilePng);
  // cv::Mat radarScanImage = imread(dataFilePng, cv::IMREAD_GRAYSCALE);
  // //
  // cv::Mat cropped;
  // cropped = cropRadarScan(radarScanImage);
  // std::string windowName = "prova taglio immagine ";
  // cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
  // cv::imshow(windowName,cropped); //show image.
  // cv::waitKey();
  //
  // std::string windowName1 = "prova taglio 2 ";
  // cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
  // cv::imshow(windowName1,radarScanImage); //show image.
  // cv::waitKey();
  //
  //
  // cv::Mat detected_edges;
  // cv::Mat edges;
  // cv::Mat finale;
  // cv::Mat finale2, finale3;
  // cv::Mat src;
  //
  //
  // finale3 = radarScanFilter(radarScanImage);
  // std::string windowNameCanny2 = "Radar Scan Image CANNY WITH Threshold ";
  // cv::namedWindow(windowNameCanny2, cv::WINDOW_AUTOSIZE);
  // cv::imshow(windowNameCanny2, finale3); //show image.
  // cv::waitKey();
  // //
  //
  // //////////////////////////////////////////////////////////////////////////////
  // // PROVA REMAPPING
  // cv::Mat cart;
  // linearPolar(cropped, cart, Point2f(cropped.cols / 2, cropped.rows / 2), 3*(cropped.rows / 2),CV_WARP_FILL_OUTLIERS | CV_INTER_LINEAR | CV_WARP_INVERSE_MAP);
  // std::string windowNameCartesian2 = "Radar Scan Image CARTESIAN ";
  // cv::namedWindow(windowNameCartesian2, cv::WINDOW_AUTOSIZE);
  // cv::imshow(windowNameCartesian2, cart); //show image.
  // cv::waitKey();
  //
  //
  // cv::Mat returnToPolar = cv::Mat::zeros(cv::Size(400,400), CV_8UC1);
  //
  // linearPolar(cart, returnToPolar, Point2f(cropped.cols / 2, cropped.rows / 2), 3*(cropped.rows / 2),CV_WARP_FILL_OUTLIERS | CV_INTER_LINEAR);
  // // std::string windowNameCartesian3 = "Radar Scan Image POLAR AGAIN ";
  // // cv::namedWindow(windowNameCartesian3, cv::WINDOW_AUTOSIZE);
  // // cv::imshow(windowNameCartesian3, returnToPolar); //show image.
  // // cv::waitKey();
  //
  // Mat cartFiltered;
  // cartFiltered = radarScanFilter(cart);
  // std::string windowNameCartesian3 = "Radar Scan Image CART FILTERED ";
  // cv::namedWindow(windowNameCartesian3, cv::WINDOW_AUTOSIZE);
  // cv::imshow(windowNameCartesian3, cartFiltered); //show image.
  // cv::waitKey();
  // // string pathProva2 = fn[1];
  // // std::cout << "prova2 "<< pathProva2 << std::endl;
  //
  // std::vector<Vector2fVector> landMarkCartesian;
  // landMarkCartesian.push_back(Vector2fVector());
  // Eigen::Vector2f scan_cartesian;
  // scan_cartesian(0) = 1;
  // scan_cartesian(1) = 4;
  // // landMarkCartesian.back().push_back(scan_cartesian);
  // landMarkCartesian.back().push_back(scan_cartesian);
  //
  // // Vector2fVector map = landMarkCartesian.(0);
  // // landMarkCartesian.at(iter).at(i)[0]
  // std::cout << "cazzo cazzo "<< landMarkCartesian.at(0).at(0)[1] << std::endl;
  //
  // Vector2fVector provaVettore;
  // Eigen::Vector2f nuovoScan;
  // nuovoScan(0) = 1;
  // nuovoScan(1) = 100;
  // provaVettore.push_back(nuovoScan);
  // std::cout << "cazzo cazzo "<< provaVettore.at(0)[1] << std::endl;
  // ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // // Prova recupero file txt
  // string dataFileTxt = "/home/luca/Documents/tesi/2019-01-10-11-46-21-radar-oxford-10k_Navtech_CTS350-X_Radar/2019-01-10-11-46-21-radar-oxford-10k/python/dataFile/1547131046353776.txt";


   return 0;
}
