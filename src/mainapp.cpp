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

  // Retrieve all the png file of the radar scan in the folder //
  string radaScanPngFilesFolder = "/home/luca/Documents/tesi/oxford_radar_robotcar_dataset_sample_small/2019-01-10-14-36-48-radar-oxford-10k-partial/radar/*.png";
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

  Eigen::Matrix<float, 2, 2> Rtot;
  Rtot << 1,0,0,1;
  Eigen::Vector2f translationVectorTot;
  translationVectorTot << 0,0;

  // Itero su tutti i radarscan della cartella
  for(int i = 0; i < 1; i++){
    dataFilePng = pathRadarScanPngFiles[i+1];
    dataFilePngSucc = pathRadarScanPngFiles[i+2];
    dataFilePngSucc2 = pathRadarScanPngFiles[i+3];
    dataBikeStrike = "/home/luca/Downloads/Bikesgray.jpg";
    dataBikeStrikePrewitt = "/home/luca/Downloads/Bikesgray_prewitt.JPG";

    cv::Mat cropped, croppedSucc, croppedSucc2;
    cv::Mat cart, cartSucc, cartSucc2;
    cv::Mat cartFiltered;
    cv::Mat locations;   // output, locations of non-zero pixels
    cv::Mat bikeStrike;
    cv::Mat prewittBike, bikeORIGINAL;


    bikeStrike = imread(dataBikeStrike, cv::IMREAD_GRAYSCALE);
    bikeORIGINAL = imread(dataBikeStrikePrewitt,  CV_32FC1);

    cv::Mat radarScanImage = imread(dataFilePng, cv::IMREAD_GRAYSCALE);
    radarScanImage = cropRadarScan(radarScanImage);




    cv::Mat radarScanImageSucc = imread(dataFilePngSucc, cv::IMREAD_GRAYSCALE);
    radarScanImageSucc = cropRadarScan(radarScanImageSucc);

    cv::Mat radarScanImageSucc2 = imread(dataFilePngSucc2, cv::IMREAD_GRAYSCALE);
    radarScanImageSucc2 = cropRadarScan(radarScanImageSucc2);

    // convert rardarscanImage in float
    radarScanImage.convertTo(radarScanImage, CV_32FC1, 1/255.0);
    radarScanImageSucc.convertTo(radarScanImageSucc, CV_32FC1, 1/255.0);
    radarScanImageSucc2.convertTo(radarScanImageSucc2, CV_32FC1, 1/255.0);
    bikeStrike.convertTo(bikeStrike, CV_32FC1, 1/255.0);



    prewittBike = prewittOperator(bikeStrike);



    // getMatrixH(prewitt, SPrime);

    // std::string sprime = "sprime  ";
    // cv::namedWindow(sprime, cv::WINDOW_AUTOSIZE);
    // cv::imshow(sprime, SPrime); //show image.
    // cv::waitKey();

    // std::string hm = "sprime  ";
    // cv::namedWindow(hm, cv::WINDOW_AUTOSIZE);
    // cv::imshow(hm, HMatrix); //show image.
    // cv::waitKey();
    // //
    ////////////////////////////////////////////////////////////////////////////
    // Vector3fVector indici;
    // indici = getIndicesOfElementsInDescendingOrder(prewitt);
    //-- Step 1: Detect the keypoints using SURF Detector
    cv::Mat provaImageSurf = imread(dataFilePng, cv::IMREAD_GRAYSCALE);

    Mat prova;
    prova =cropRadarScan(provaImageSurf, 11, 0, 2000, 400);

    double maxRadius = 400.0;
    Point2f center( 400, 400);
    int flags = INTER_LINEAR + WARP_FILL_OUTLIERS + WARP_INVERSE_MAP;
    warpPolar(provaImageSurf, cart, Size(800,800) , center, maxRadius,  flags);

    Mat  provaBlur, provaBlur2,cartBlur, cartBlurSucc, cartBlurSucc2;
    Mat provaBlura,provaBlurb,provaBlurc,provaBlurd;
    // blur( cart, cartBlur, Size(3,3) );
    GaussianBlur(cart,cartBlur,Size(3,3),0);
    // GaussianBlur(cart,provaBlura,Size(3,3),0);
    // GaussianBlur(provaBlura,provaBlurc,Size(3,3),0);
    // GaussianBlur(provaBlurc,provaBlur,Size(3,3),0);
    // GaussianBlur(provaBlur,cartBlur,Size(3,3),0);

    // const String window_name = "Sobel Demo - Simple Edge Detector";
    // int ksize = 3;
    // // int ksize = CV_SCHARR;
    // int scale = 1;
    // int delta = 0;
    // int ddepth = CV_16S;
    // Mat grad_x, grad_y;
    // Mat abs_grad_x, abs_grad_y;
    // Mat grad, grad2;
    // Sobel(provaBlura, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    // Sobel(provaBlura, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
    // // converting back to CV_8U
    // convertScaleAbs(grad_x, abs_grad_x);
    // convertScaleAbs(grad_y, abs_grad_y);
    // addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, cartBlur);



    //
    // int minHessian = 1500;
    // Ptr<SURF> detector = SURF::create( minHessian );
    // std::vector<KeyPoint> keypoints;
    // detector->detect( cart, keypoints );
    // //-- Draw keypoints
    // Mat img_keypoints;
    // drawKeypoints( cart, keypoints, img_keypoints );
    // //-- Show detected (drawn) keypoints
    // imshow("SURF Keypoints", img_keypoints );
    //
    //
    cv::Mat provaImageSurf2 = imread(dataFilePngSucc, cv::IMREAD_GRAYSCALE);
    Mat prova2;
    prova2 =cropRadarScan(provaImageSurf2, 11, 0, 2000, 400);
    warpPolar(provaImageSurf2, cartSucc, Size(800,800) , center, maxRadius,  flags);
    GaussianBlur(cartSucc,cartBlurSucc,Size(3,3),0);

    cv::Mat provaImageSurf3 = imread(dataFilePngSucc2, cv::IMREAD_GRAYSCALE);
    Mat prova3;
    prova3 =cropRadarScan(provaImageSurf3, 11, 0, 2000, 400);
    warpPolar(provaImageSurf3, cartSucc2, Size(800,800) , center, maxRadius,  flags);
    GaussianBlur(cartSucc2,cartBlurSucc2,Size(3,3),0);
    // blur( cartSucc, cartBlurSucc, Size(3,3) );

    // GaussianBlur(cartSucc,provaBlurb,Size(3,3),0);
    // GaussianBlur(provaBlurb,provaBlurd,Size(3,3),0);
    // GaussianBlur(provaBlurd,provaBlur2,Size(3,3),0);
    // GaussianBlur(provaBlur2,cartBlurSucc,Size(3,3),0);

    // Mat grad_x2, grad_y2;
    // Mat abs_grad_x2, abs_grad_y2;
    //
    // Sobel(provaBlurb, grad_x2, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    // Sobel(provaBlurb, grad_y2, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
    // // converting back to CV_8U
    // convertScaleAbs(grad_x2, abs_grad_x2);
    // convertScaleAbs(grad_y2, abs_grad_y2);
    // addWeighted(abs_grad_x2, 0.5, abs_grad_y2, 0.5, 0, cartBlurSucc);
    //
    //
    // Ptr<SURF> detector2 = SURF::create( minHessian );
    // std::vector<KeyPoint> keypoints2;
    // detector2->detect( cartSucc , keypoints2 );
    // //-- Draw keypoints
    // Mat img_keypoints2;
    // drawKeypoints( cart, keypoints2, img_keypoints2 );
    // //-- Show detected (drawn) keypoints
    // imshow("SURF Keypoints2", img_keypoints2 );
    // waitKey();
    // return 0;
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    // int minHessian = 1700;
    // Ptr<SURF> detector = SURF::create( minHessian );
    // std::vector<KeyPoint> keypoints1, keypoints2;
    // Mat descriptors1, descriptors2;
    // detector->detectAndCompute( cart, noArray(), keypoints1, descriptors1 );
    // detector->detectAndCompute( cartSucc, noArray(), keypoints2, descriptors2 );
    // //-- Step 2: Matching descriptor vectors with a brute force matcher
    // // Since SURF is a floating-point descriptor NORM_L2 is used
    // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    // std::vector< DMatch > matches;
    // matcher->match( descriptors1, descriptors2, matches );
    // // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    // // std::vector< DMatch > matches;
    // // matcher->match( descriptors1, descriptors2, matches );
    // //-- Draw matches
    // Mat img_matches;
    // drawMatches( cart, keypoints1, cartSucc, keypoints2, matches, img_matches );
    // // drawMatches( cart, keypoints1, cart, keypoints1, matches, img_matches );
    // //-- Show detected matches
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Initiate ORB detector
    // Ptr<FeatureDetector> detectorORB = ORB::create();
    //
    // // find the keypoints and descriptors with ORB
    // std::vector<KeyPoint> keypoints_object, keypoints_scene;
    // detectorORB->detect(cartBlur, keypoints_object);
    // detectorORB->detect(cartBlurSucc, keypoints_scene);
    //
    // Ptr<DescriptorExtractor> extractor = ORB::create();
    // Mat descriptors_object, descriptors_scene;
    // extractor->compute(cartBlur, keypoints_object, descriptors_object );
    // extractor->compute(cartBlurSucc, keypoints_scene, descriptors_scene );
    //
    // //-- Draw keypoints
    // Mat img_keypointsORB;
    // drawKeypoints( cartBlur, keypoints_object, img_keypointsORB );
    // //-- Show detected (drawn) keypoints
    // imshow("ORB Keypoints", img_keypointsORB );
    // waitKey();
    //
    // Mat img_keypointsORB2;
    // drawKeypoints( cartBlurSucc, keypoints_scene, img_keypointsORB2 );
    // //-- Show detected (drawn) keypoints
    // imshow("ORB Keypoints2", img_keypointsORB2 );
    // waitKey();
    //
    // Ptr<DescriptorMatcher> matcherORB = DescriptorMatcher::create("BruteForce-Hamming");
    // std::vector< std::vector<DMatch> > knn_matchesORB;
    // matcherORB->knnMatch( descriptors_object, descriptors_scene, knn_matchesORB, 2 );
    //
    //
    // const float ratio_threshORB = 0.7f;
    // std::vector<DMatch> good_matchesORB;
    // for (size_t i = 0; i < knn_matchesORB.size(); i++)
    // {
    //     if (knn_matchesORB[i][0].distance < ratio_threshORB * knn_matchesORB[i][1].distance)
    //     {
    //         good_matchesORB.push_back(knn_matchesORB[i][0]);
    //     }
    // }
    //
    // //-- Draw matches
    // Mat img_matchesORB3;
    // // drawMatches( cart, keypoints1, cartSucc, keypoints2, good_matches, img_matches, Scalar::all(-1),
    // //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // drawMatches( cartBlur, keypoints_object, cartBlurSucc, keypoints_scene, good_matchesORB, img_matchesORB3, Scalar::all(-1),
    //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // imshow("Matches", img_matchesORB3 );
    // waitKey();

    // BFMatcher matcherORB(NORM_L2);
    // std::vector<vector<DMatch> > matchesORB;
    // matcher.knnMatch(descriptors1, descriptors2, matches,2);
    //
    // std::vector<DMatch> match1;
    // std::vector<DMatch> match2;
    //
    // for(int i=0; i<matches.size(); i++)
    // {
    //     match1.push_back(matches[i][0]);
    //     match2.push_back(matches[i][1]);
    // }
    //
    // Mat img_matches1, img_matches2;
    // drawMatches(img1, kp1, img2, kp2, match1, img_matches1);
    // drawMatches(img1, kp1, img2, kp2, match2, img_matches2);
    // Ptr<FeatureDetector> detectorORB = ORB::create();
    // // OrbFeatureDetector detectorORB //OrbFeatureDetector detector;SurfFeatureDetector
    // vector<KeyPoint> keypointsORB1;
    // detectorORB.detect(cartBlur, keypointsORB1);
    // vector<KeyPoint> keypointsORB2;
    // detectorORB.detect(cartBlurSucc, keypointsORB2);
    //
    // OrbDescriptorExtractor extractorORB; //OrbDescriptorExtractor extractor; SurfDescriptorExtractor extractor;
    // Mat descriptors_ORB1, descriptors_ORB2;
    // extractorORB.compute( cartBlur, keypointsORB1, descriptors_ORB1 );
    // extractorORB.compute( cartBlurSucc, keypointsORB2, descriptors_ORB2 );
    //
    // //-- Step 3: Matching descriptor vectors with a brute force matcher
    // BFMatcher matcher(NORM_L2, true);   //BFMatcher matcher(NORM_L2);
    //
    // vector< DMatch> matches;
    // matcher.match(descriptors_ORB1, descriptors_ORB2, matches);





    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 300;
    Ptr<SURF> detector = SURF::create( minHessian );
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
    // -- Draw keypoints
    // Mat mapPoints1;
    // drawKeypoints( cartBlur, keypoints1, mapPoints1 );
    // //-- Show detected (drawn) keypoints
    // imshow("First map 1", mapPoints1 );
    // waitKey();

    LocalMap map;
    map.initFirstMap(keypoints1,descriptors1);

    //-- Draw matches
    // Mat img_matches;
    // // drawMatches( cart, keypoints1, cartSucc, keypoints2, good_matches, img_matches, Scalar::all(-1),
    // //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // drawMatches( cartBlur, keypoints1, cartBlurSucc, keypoints2, good_matches, img_matches, Scalar::all(-1),
    //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // imshow("Matches", img_matches );
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

    Eigen::Matrix<float, 2, 2> Rf;
    Rf = rigidBodyMotionSurf(keypoints1, keypoints2, ultimate_matches);
    Rtot = Rtot*Rf;


    Eigen::Vector2f mean1;
    mean1 = meanScanSurf1(keypoints1, ultimate_matches);
    Eigen::Vector2f mean2;
    mean2 = meanScanSurf2(keypoints2, ultimate_matches);

    Eigen::Vector2f translationVectorf;
    translationVectorf = mean2 - Rf * mean1;
    translationVectorTot += translationVectorf;
    std::cerr << "translation vector 1 "<< translationVectorTot << '\n';


    // map.trackLocalMap(keypoints2,descriptors2,ultimate_matches,Rtot,translationVectorTot);

    //-- Draw final matches
    // Mat img_matches2;
    // // drawMatches( cartBlur, keypoints1, cartBlurSucc, keypoints2, ultimate_matches, img_matches2, Scalar::all(-1),
    // //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // drawMatches( cartBlur, keypoints1, cartBlurSucc, keypoints2, ultimate_matches, img_matches2, Scalar::all(-1),
    //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // imshow("Matches ultimate ", img_matches2 );
    // waitKey();

    /////////////////////////////////////////////////////////////////////////////
    // Track third scan for localMap try
    Ptr<SURF> detector2 = SURF::create( minHessian );
    detector2->detectAndCompute( cartBlurSucc2, noArray(), keypoints3, descriptors3 );
    Ptr<DescriptorMatcher> matcher2 = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches2;
    matcher2->knnMatch( descriptors2, descriptors3, knn_matches2, 2 );
    std::vector<DMatch> good_matches2;
    for (size_t i = 0; i < knn_matches2.size(); i++)
    {
        if (knn_matches2[i][0].distance < ratio_thresh * knn_matches2[i][1].distance)
        {
            good_matches2.push_back(knn_matches2[i][0]);
        }
    }
    std::vector<int> maxClique2;
    maxClique2 = createConsistencyMatrix(keypoints2, keypoints3, good_matches2);

    std::vector<DMatch> ultimate_matches2;
    for (size_t i = 0; i < good_matches2.size(); i++) {
      if (maxClique2[i] == 1) {
        ultimate_matches2.push_back(good_matches2[i]);
      }
    }
    //-- Draw matches
    Mat img_matches;
    // drawMatches( cart, keypoints1, cartSucc, keypoints2, good_matches, img_matches, Scalar::all(-1),
    //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    drawMatches( cartBlurSucc, keypoints2, cartBlurSucc2, keypoints3, ultimate_matches2, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imshow("Matches", img_matches );
    waitKey();

    std::cerr << "ultimate_matches2 main "<< ultimate_matches2.size() << '\n';
    Rf = rigidBodyMotionSurf(keypoints2, keypoints3, ultimate_matches2);
    std::cerr << "Rf "<< Rf << '\n';
    Rtot = Rtot*Rf;
    mean1 = meanScanSurf1(keypoints2, ultimate_matches2);
    mean2 = meanScanSurf2(keypoints3, ultimate_matches2);
    translationVectorf = mean2 - Rf * mean1;
    translationVectorTot += translationVectorf;
    std::cerr << "translation vector 2 "<< translationVectorTot << '\n';
    std::cerr << "R tot main "<< Rtot << '\n';

    map.trackLocalMap(keypoints3,descriptors3, Rtot, translationVectorTot);
    map.dispMap();
    /////////////////////////////////////////////////////////////////////////////

    RGBImage local_image(img_width, img_height);
    local_image.create(img_width, img_height);
    local_image=cv::Vec3b(255,255,255);

    Vector2fVector scan_for_disp;
    for (int i=0; i<ultimate_matches.size(); i++){
      Eigen::Vector2f cartesian_point(keypoints1[ultimate_matches[i].queryIdx].pt.x, keypoints1[ultimate_matches[i].queryIdx].pt.y);
      scan_for_disp.push_back(cartesian_point);
    }
    drawPoints(local_image, scan_for_disp, cv::Scalar(255,0,0),1);

    Vector2fVector post_scan_for_disp;
    for (int i=0; i<ultimate_matches.size(); i++){
      Eigen::Vector2f cartesian_point(keypoints2[ultimate_matches[i].trainIdx].pt.x, keypoints2[ultimate_matches[i].trainIdx].pt.y);
      post_scan_for_disp.push_back(cartesian_point);
    }
    drawPoints(local_image, post_scan_for_disp, cv::Scalar(0,255,0),1);

    Vector2fVector transformed_scan_for_disp;
    Eigen::Vector2f positionLand;
    for (int i=0; i<ultimate_matches.size(); i++){
      // Eigen::Vector2f transformed_point = init_transform.inverse()*scans_cartesian.at(iter).at(i);

      positionLand(0) = keypoints1[ultimate_matches[i].queryIdx].pt.x;
      positionLand(1) = keypoints1[ultimate_matches[i].queryIdx].pt.y;
      Eigen::Vector2f transformed_point = Rf * positionLand + translationVectorf;

      Eigen::Vector2f cartesian_point(transformed_point(0), transformed_point(1));
      transformed_scan_for_disp.push_back(cartesian_point);

    }
    // 0,150,255
    drawPoints(local_image, transformed_scan_for_disp, cv::Scalar(0,0,255),1);

    cv::imshow("Scan matcher", local_image);

    waitKey(0);
    return 0;
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    int lmax = 1000;
    // key1 = keyPointExtraction(bikeStrike, lmax);
    // key2 = keyPointExtraction(bikeORIGINAL, lmax);
    // radarScanImage =  cropRadarScan(radarScanImage, 0, 0, 1750 ,400);
    // radarScanImageSucc =  cropRadarScan(radarScanImageSucc, 0, 0, 1750,400);

    key1 = keyPointExtraction(radarScanImage, lmax);
    key2 = keyPointExtraction(radarScanImageSucc, lmax);
    // con lmax=200 ci sono più keypoint nel secondo scan, quindi devo invertire key1 con key2
    // key2 = keyPointExtraction(radarScanImage, lmax);
    // key1 = keyPointExtraction(radarScanImageSucc, lmax);
    // RICORDA DI INSERIRE UN CHECK SU NUMERO DI LANMARK DEL PRIMO E SECONDO SCAN
    VectorOfDescriptorVector provaDescrittore, provaDescrittore2, provaDescrittoreRelaxation;

    provaDescrittore = createDescriptor(key1);
    //////////////////////////////////////////////////////////////////////////////
    // Define random generator with Gaussian distribution
    const double mean = 0.0;
    const double stddev = 0.2;
    std::default_random_engine generator;
    std::normal_distribution<double> dist(mean, stddev);

    provaDescrittoreRelaxation = provaDescrittore;
    float x1,y1;
    float traslazione = 10;
    // for (size_t i = 0; i < provaDescrittore.size(); i++) {
    //   x1 = provaDescrittore[i](0)*cos(3.14/8) + provaDescrittore[i](1)*(-sin(3.14/8))+ traslazione;
    //   y1 = provaDescrittore[i](0)*sin(3.14/8) + provaDescrittore[i](1)*cos(3.14/8) + traslazione;
    //   provaDescrittoreRelaxation[i](0) = x1;
    //   provaDescrittoreRelaxation[i](1) = y1;
    //   for (size_t j = 2; j < 400+3500; j++) {
    //     provaDescrittoreRelaxation[i](j) += (float) dist(generator);
    //   }
    // }
    ////////////////////////////////////////////////////////////////////////////
    provaDescrittore2 = createDescriptor(key2);
    Eigen::Matrix<float, 3, Eigen::Dynamic> provaMatchProposal;
    // provaMatchProposal = matchProposal(provaDescrittore,provaDescrittore);
    // provaMatchProposal = matchProposal(provaDescrittore,provaDescrittoreRelaxation);
    provaMatchProposal = matchProposal(provaDescrittore,provaDescrittore2);
    // for (size_t i = 0; i < 284; i++) {
    //   std::cerr << "proposal "<< provaMatchProposal(1,i) << '\n';
    // }
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> pairWiseMatrix;
    // int numberOfLandmarks = provaDescrittore.size();
    // pairWiseMatrix.resize(numberOfLandmarks, numberOfLandmarks);

    // pairWiseMatrix = createPairwiseCompatibilities(provaDescrittore,provaDescrittore, provaMatchProposal);
    // pairWiseMatrix = createPairwiseCompatibilities(provaDescrittore,provaDescrittoreRelaxation, provaMatchProposal);
    pairWiseMatrix = createPairwiseCompatibilities(provaDescrittore,provaDescrittore2, provaMatchProposal);

    // pairWiseMatrix = createPairwiseCompatibilities(provaDescrittore,provaDescrittore2, provaMatchProposal);
    // createPairwiseCompatibilities(provaDescrittore,provaDescrittore2, provaMatchProposal);
    // provaDescrittore2 =createDescriptor(key2)
    Eigen::Matrix<float, 1, Eigen::Dynamic> optimizedAssociationSolution;
    optimizedAssociationSolution = greedyAlgorithm(pairWiseMatrix, provaMatchProposal);
    // for (size_t i = 0; i < 3500+400; i++) {
    //   std::cerr << "descrittore "<< provaDescrittore[3](i) << '\n';
    //   if (i == 400) {
    //     std::cerr << "angular descriptor finished" << '\n';
    //   }
    // }
    // std::cerr << "descrittore dimensione  "<< provaDescrittore.size() << '\n';
    ////////////////////////////////////////////////////////////////////////////
    // Prova metodo callis con rotazione prestabilita ma non unitaria
    // provaDescrittoreRelaxation = provaDescrittore;
    // float x1,y1;
    // float traslazione = 0;
    // for (size_t i = 0; i < provaDescrittore.size(); i++) {
    //   x1 = provaDescrittore[i](0)*(1/sqrt(2)) + provaDescrittore[i](1)*(-1/sqrt(2))+ traslazione;
    //   y1 = provaDescrittore[i](0)*(1/sqrt(2)) + provaDescrittore[i](1)*(1/sqrt(2)) + traslazione;
    //   provaDescrittoreRelaxation[i](0) = x1;
    //   provaDescrittoreRelaxation[i](1) = y1;
    // }
    // Eigen::Matrix<float, 2,2> rotationMatrixRProva;
    // rotationMatrixRProva = rigidBodyMotion(provaDescrittore, provaDescrittoreRelaxation, provaMatchProposal, optimizedAssociationSolution);
    //  std::cerr << "rotationMatrixRprova "<< rotationMatrixRProva << '\n';
    //  std::cerr << "inversa soluzione "<< rotationMatrixRProva.inverse() << '\n';
    ////////////////////////////////////////////////////////////////////////////
    std::cerr << "ROTAZIONE DUE" << '\n';
    Eigen::Matrix<float, 2,2> rotationMatrixR;
    // rotationMatrixR = rigidBodyMotion(provaDescrittore, provaDescrittore, provaMatchProposal, optimizedAssociationSolution);
    // rotationMatrixR = rigidBodyMotion(provaDescrittore, provaDescrittoreRelaxation, provaMatchProposal, optimizedAssociationSolution);
    rotationMatrixR = rigidBodyMotion(provaDescrittore, provaDescrittore2, provaMatchProposal, optimizedAssociationSolution);
    std::cerr << "rotationMatrixR "<< rotationMatrixR << '\n';

    // Eigen::Vector2f meanScan1;
    // meanScan1 << 0,0;
    // Eigen::Vector2f meanScan2;
    // meanScan2 << 0,0;

    ////////////////////////////////////////////////////////////////////////////
    // for (int i = 0; i < provaDescrittore.size(); i++) {
    //   meanScan1(0) += provaDescrittore[i](0);
    //   meanScan1(1) += provaDescrittore[i](1);
    //
    //   meanScan2(0) += provaDescrittore2[(int)provaMatchProposal(1,i)](0);
    //   meanScan2(1) += provaDescrittore2[(int)provaMatchProposal(1,i)](1);
    // }
    // meanScan1 = meanScan1 / (int)provaDescrittore.size();
    // meanScan2 = meanScan2 / (int)provaDescrittore.size();
    ////////////////////////////////////////////////////////////////////////////
    Eigen::Vector4f meanScan1And2;
    meanScan1And2 = meanScanWithAssociation(provaDescrittore,
                                            provaDescrittore2,
                                            provaMatchProposal,
                                            optimizedAssociationSolution);
    // meanScan1And2 = meanScanWithAssociation(provaDescrittore,
    //                                         provaDescrittoreRelaxation,
    //                                         provaMatchProposal,
    //                                         optimizedAssociationSolution);
    ////////////////////////////////////////////////////////////////////////////
    // Eigen::Vector2f translationVector = meanScan2 - rotationMatrixR * meanScan1;
    // Eigen::Vector2f meanScan1;
    // meanScan1 << meanScan1And2(0),meanScan1And2(1);
    // Eigen::Vector2f meanScan2;
    // meanScan2 << meanScan1And2(2),meanScan1And2(3);
    // Eigen::Vector2f translationVector;
    // std::cerr << "arrivato " << '\n';
    // translationVector = meanScan2 - rotationMatrixR * meanScan1;
    // std::cerr << "MEAN1 "<< meanScan1 << '\n';
    // std::cerr << "MEAN2 "<< meanScan2 << '\n';
    // std::cerr << "translation VECTOR " <<translationVector << '\n';
    // std::cerr << "determinant "<< rotationMatrixR.determinant() << '\n';
    // std::cerr << "descrittore dimensione  2 "<< provaDescrittore2.size() << '\n';
    // std::cout << "size landmark matrix"<< key1.size() << '\n';
    // key3 = keyPointExtraction(radarScanImageSucc2, lmax);

    // keyPointExtraction(radarScanImageSucc, lmax);
    // cropped = cropRadarScan(radarScanImage);
    // croppedSucc = cropRadarScan(radarScanImageSucc);

    // double maxRadius = 400.0;
    // Point2f center( 400, 400);
    // int flags = INTER_LINEAR + WARP_FILL_OUTLIERS + WARP_INVERSE_MAP;
    //
    // warpPolar(radarScanImage, cart, Size(800,800) , center, maxRadius,  flags);

    // warpPolar(cropped, cart, Size(800,800) , center, maxRadius,  flags);
    // cv::linearPolar(cropped, cart, Point2f(cropped.cols / 2, cropped.rows / 2), (cropped.rows / 2), CV_INTER_LINEAR | CV_WARP_INVERSE_MAP);
    // cv::linearPolar(croppedSucc, cartSucc, Point2f(croppedSucc.cols / 2, croppedSucc.rows / 2), (croppedSucc.rows / 2), CV_INTER_LINEAR | CV_WARP_INVERSE_MAP);
    // cart = cropRadarScan(cart, 1550, 0, 400, 400);
    // cartSucc = cropRadarScan(cartSucc, 1550, 0, 400, 400);
    // // linearPolar(radarScanImage, cart, Point2f(radarScanImage.cols / 2, radarScanImage.rows / 2), 3*(radarScanImage.rows / 2),CV_WARP_FILL_OUTLIERS | CV_INTER_LINEAR | CV_WARP_INVERSE_MAP);

    // PROVA SIFT
    // cv::SiftFeatureDetector detector;
    // std::vector<cv::KeyPoint> keypoints;
    // detector.detect(cart, keypoints);
    //
    // // Add results to image and save.
    // cv::Mat output;
    // cv::drawKeypoints(cart, keypoints, output);
    // cv::imwrite("sift_result.jpg", output);
    //

    // SIFT non più supportato da opencv?
    // Prova SURF
    //-- Step 1: Detect the keypoints using SURF Detector
    // int minHessian = 700;
    // Ptr<SURF> detector = SURF::create( minHessian );
    // std::vector<KeyPoint> keypoints;
    // detector->detect( cart, keypoints );
    //
    // //-- Draw keypoints
    // Mat img_keypoints;
    // drawKeypoints( cart, keypoints, img_keypoints );
    // //-- Show detected (drawn) keypoints
    // imshow("SURF Keypoints", img_keypoints );
    // waitKey();

    // Prova confronto scan SURF (bruteforce matching)
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    // int minHessian = 800;
    // Ptr<SURF> detector = SURF::create( minHessian );
    // std::vector<KeyPoint> keypoints1, keypoints2;
    // Mat descriptors1, descriptors2;
    // detector->detectAndCompute( cart, noArray(), keypoints1, descriptors1 );
    // detector->detectAndCompute( cartSucc, noArray(), keypoints2, descriptors2 );
    // //-- Step 2: Matching descriptor vectors with a brute force matcher
    // // Since SURF is a floating-point descriptor NORM_L2 is used
    // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    // std::vector< DMatch > matches;
    // matcher->match( descriptors1, descriptors2, matches );
    // //-- Draw matches
    // Mat img_matches;
    // drawMatches( cart, keypoints1, cartSucc, keypoints2, matches, img_matches );
    // //-- Show detected matches
    // imshow("Matches", img_matches );
    // waitKey();
    //

    cartFiltered = radarScanFilter(cart);
    // cartFiltered.resize(1000);
    // A questo punti la mia immagini in coordinate cartesiane
    // è composta da punti i cui pixel hanno valore 255 e 0. Devo identificare le coordinate di quelli con valore 255
    // cv::findNonZero(cartFiltered, locations);
    // int numberOfPoint = locations.rows; // Number of landmarks
    // // Now i iterate over the number of landMark
    // landMarkCartesian.push_back(Vector2fVector());
    // for(int k = 0; k < numberOfPoint; k++){
    //   pnt = locations.at<Point>(k);
    //   // Converto coordinate pixel in coordinate world
    //   pixelRange = (cropped.rows / 2);
    //   pntWorld = pixelToWorldCoord(pnt, pixelRange);
    //   //
    //   landMarkCartesian.back().push_back(pntWorld);
    //
    //
    //
    // }
    //
    // std::string landmarkprint1 = " landmark scan1 ";
    // cv::namedWindow(landmarkprint1, cv::WINDOW_AUTOSIZE);
    // cv::imshow(landmarkprint1, key1); //show image.
    // cv::waitKey();

    std::cerr << "bla bla bla " << '\n';
    // std::cerr << "optimizedAssociationSolution "<< optimizedAssociationSolution << '\n';
    // RGBImage local_image(img_width, img_height);
    // local_image.create(img_width, img_height);
    // local_image=cv::Vec3b(255,255,255);
    // // Global view
    // RGBImage map_image(img_width, img_height);
    // map_image.create(img_width, img_height);
    // map_image=cv::Vec3b(255,255,255);
    //
    // RGBImage refined_map_image(img_width, img_height);
    // refined_map_image.create(img_width, img_height);
    // refined_map_image=cv::Vec3b(255,255,255);
    //
    // // Show 2 scans
    // // Scale and translate scans for better display
    // std::cerr << "translationVector "<<translationVector << '\n';
    // Vector2fVector scan_for_disp;
    // int numberOfLandmarks = provaDescrittore.size(); // numero di landmark nel radarscan "iter"
    // std::cout << "landmarks nello scan di partenza "<< numberOfLandmarks  << std::endl;
    ////////////////////////////////////////////////////////////////////////////
    // for (int i=0; i<numberOfLandmarks; i++){
    //   Eigen::Vector2f cartesian_point(provaDescrittore[i](0)*0.3 + img_width/2, provaDescrittore[i](1)*0.3 + img_height/2);
    //   scan_for_disp.push_back(cartesian_point);
    // }
    // drawPoints(local_image, scan_for_disp, cv::Scalar(255,0,0),1);
    //
    // Vector2fVector prev_scan_for_disp;
    // int numberOfLandmarks2 = provaDescrittore2.size();
    // for (int i=0; i<numberOfLandmarks2; i++){
    //   Eigen::Vector2f cartesian_point(provaDescrittore2[i](0)*0.3 + img_width/2, provaDescrittore2[i](1)*0.3 + img_height/2);
    //   prev_scan_for_disp.push_back(cartesian_point);
    // }
    // drawPoints(local_image, prev_scan_for_disp, cv::Scalar(0,255,0),1);
    ////////////////////////////////////////////////////////////////////////////
    // std::cerr << "PROVA MATCH proposal "<<  provaMatchProposal << '\n';
    // Vector2fVector prev_scan_for_disp;
    // IntPairVector correspondences;
    // int numberOfLandmarks2 = provaDescrittore2.size();
    // int indiceProvaAssociazioni;
    //
    // for (int i=0; i<numberOfLandmarks; i++){
    //   if (optimizedAssociationSolution[i] == 1) {
    //     Eigen::Vector2f cartesian_point(provaDescrittore[i](0)*0.2 + img_width/2, provaDescrittore[i](1)*0.2 + img_height/2);
    //     scan_for_disp.push_back(cartesian_point);
    //
    //     Eigen::Vector2f cartesian_point2(provaDescrittore2[(int)provaMatchProposal(1,i)](0)*0.2 + img_width/2, provaDescrittore2[(int)provaMatchProposal(1,i)](1)*0.2 + img_height/2);
    //     // Eigen::Vector2f cartesian_point2(provaDescrittore[(int)provaMatchProposal(1,i)](0)*0.3 + img_width/2, provaDescrittore[(int)provaMatchProposal(1,i)](1)*0.3 + img_height/2);
    //     // Eigen::Vector2f cartesian_point2(provaDescrittoreRelaxation[(int)provaMatchProposal(1,i)](0)*0.3 + img_width/2, provaDescrittoreRelaxation[(int)provaMatchProposal(1,i)](1)*0.3 + img_height/2);
    //     prev_scan_for_disp.push_back(cartesian_point2);
    //     indiceProvaAssociazioni = prev_scan_for_disp.size();
    //
    //     IntPair correspondence;
    //     correspondence.first = indiceProvaAssociazioni-1;
    //     correspondence.second = indiceProvaAssociazioni-1;
    //     correspondences.push_back(correspondence);
    //   }
    // }
    //
    // drawPoints(local_image, scan_for_disp, cv::Scalar(255,0,0),1);
    // drawPoints(local_image, prev_scan_for_disp, cv::Scalar(0,0,255),1);
    // drawCorrespondences(local_image, prev_scan_for_disp, scan_for_disp, correspondences, cv::Scalar(0,255,0));
    // // Draw result in local view
    // Vector2fVector transformed_scan_for_disp;
    // Eigen::Vector2f positionLand;
    // for (int i=0; i<numberOfLandmarks; i++){
    //
    //   // Eigen::Vector2f transformed_point = init_transform.inverse()*scans_cartesian.at(iter).at(i);
    //   if (optimizedAssociationSolution[i] == 1 ) {
    //     positionLand(0) = provaDescrittore[i](0);
    //     positionLand(1) = provaDescrittore[i](1);
    //     Eigen::Vector2f transformed_point = rotationMatrixR * positionLand + translationVector;
    //
    //     Eigen::Vector2f cartesian_point(transformed_point(0)*0.2 + img_width/2, transformed_point(1)*0.2 + img_height/2);
    //     transformed_scan_for_disp.push_back(cartesian_point);
    //   }
    // }
    // // 0,150,255
    // drawPoints(local_image, transformed_scan_for_disp, cv::Scalar(0,255,250),1);

    // CORRESPONDENCES
    // IntPairVector correspondences;
    // for (size_t i = 0; i < provaDescrittore.size(); i++) {
    //   IntPair correspondence;
    //   correspondence.first = i;
    //   correspondence.second = (int)provaMatchProposal(1,i);
    //   correspondences.push_back(correspondence);
    // }
    // drawCorrespondences(local_image, prev_scan_for_disp, scan_for_disp, correspondences, cv::Scalar(0,255,0));


    // cv::imshow("Scan matcher", local_image);
    // cv::waitKey(0);
  }
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
