#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
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
  // Itero su tutti i radarscan della cartella
  for(int i = 0; i < 1; i++){
    dataFilePng = pathRadarScanPngFiles[i];
    dataFilePngSucc = pathRadarScanPngFiles[i+1];
    dataFilePngSucc2 = pathRadarScanPngFiles[i+2];
    dataBikeStrike = "/home/luca/Downloads/Bikesgray.jpg";
    dataBikeStrikePrewitt = "/home/luca/Downloads/Bikesgray_prewitt.JPG";

    cv::Mat cropped, croppedSucc, croppedSucc2;
    cv::Mat cart, cartSucc;
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


    // Prova prewittOperator
    Mat_<float> prewitt;
    prewitt = prewittOperator(radarScanImage);

    std::string scanPrewittPolar = "scanPrewittPolar ";
    cv::namedWindow(scanPrewittPolar, cv::WINDOW_AUTOSIZE);
    cv::imshow(scanPrewittPolar, prewitt); //show image.
    cv::waitKey();


    // PRINT BICLICLETTE PREWITT ////////////////////////////////////////
    // std::string bho = "prewitt bho ";
    // cv::namedWindow(bho, cv::WINDOW_AUTOSIZE);
    // cv::imshow(bho, bikeORIGINAL); //show image.
    // cv::waitKey();
    //
    // std::string bikebike = "prewitt  ";
    // cv::namedWindow(bikebike, cv::WINDOW_AUTOSIZE);
    // cv::imshow(bikebike, prewittBike); //show image.
    // cv::waitKey();
    ///////////////////////////////////////////////////////////////////
    // std::string radar = "scan  ";
    // cv::namedWindow(radar, cv::WINDOW_AUTOSIZE);
    // cv::imshow(radar, radarScanImage); //show image.
    // cv::waitKey();


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

    // Vector3fVector indici;
    // indici = getIndicesOfElementsInDescendingOrder(prewitt);


    int lmax = 399;
    // key1 = keyPointExtraction(bikeStrike, lmax);
    // key2 = keyPointExtraction(bikeORIGINAL, lmax);

    key1 = keyPointExtraction(radarScanImage, lmax);
    // key2 = keyPointExtraction(radarScanImageSucc, lmax);
    createDescriptor(key1);
    std::cerr << "porco cazzo" << '\n';
    // std::cout << "size landmark matrix"<< key1.size() << '\n';
    // key3 = keyPointExtraction(radarScanImageSucc2, lmax);

    // keyPointExtraction(radarScanImageSucc, lmax);
    // cropped = cropRadarScan(radarScanImage);
    // croppedSucc = cropRadarScan(radarScanImageSucc);

    double maxRadius = 400.0;
    Point2f center( 400, 400);
    int flags = INTER_LINEAR + WARP_FILL_OUTLIERS + WARP_INVERSE_MAP;

    warpPolar(radarScanImage, cart, Size(800,800) , center, maxRadius,  flags);

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
    cv::findNonZero(cartFiltered, locations);
    int numberOfPoint = locations.rows; // Number of landmarks
    // Now i iterate over the number of landMark
    landMarkCartesian.push_back(Vector2fVector());
    for(int k = 0; k < numberOfPoint; k++){
      pnt = locations.at<Point>(k);
      // Converto coordinate pixel in coordinate world
      pixelRange = (cropped.rows / 2);
      pntWorld = pixelToWorldCoord(pnt, pixelRange);
      //
      landMarkCartesian.back().push_back(pntWorld);



    }

    std::string landmarkprint1 = " landmark scan1 ";
    cv::namedWindow(landmarkprint1, cv::WINDOW_AUTOSIZE);
    cv::imshow(landmarkprint1, key1); //show image.
    cv::waitKey();

    // std::string landmarkprint2 = "landmark scan2  ";
    // cv::namedWindow(landmarkprint2, cv::WINDOW_AUTOSIZE);
    // cv::imshow(landmarkprint2, key2); //show image.
    // cv::waitKey();

    // std::string landmarkprint3 = "landmark scan3  ";
    // cv::namedWindow(landmarkprint3, cv::WINDOW_AUTOSIZE);
    // cv::imshow(landmarkprint3, key3); //show image.
    // cv::waitKey();
    // std::string matriceH = "Normale  ";
    // cv::namedWindow(matriceH, cv::WINDOW_AUTOSIZE);
    // cv::imshow(matriceH, HMatrix); //show image.
    // cv::waitKey();
    //
    //
    // std::string windowNameCanny1 = "cartesian point  ";
    // cv::namedWindow(windowNameCanny1, cv::WINDOW_AUTOSIZE);
    // cv::imshow(windowNameCanny1, cart); //show image.
    // cv::waitKey();
    //
    // std::string windowNameCanny2 = "cartesian point  ";
    // cv::namedWindow(windowNameCanny2, cv::WINDOW_AUTOSIZE);
    // cv::imshow(windowNameCanny2, cartFiltered); //show image.
    // cv::waitKey();

    // std::string landmarkprint1 = " landmark scan1 ";
    // cv::namedWindow(landmarkprint1, cv::WINDOW_AUTOSIZE);
    // cv::imshow(landmarkprint1, key1); //show image.
    // cv::waitKey();
    //
    // std::string landmarkprint2 = "landmark scan2  ";
    // cv::namedWindow(landmarkprint2, cv::WINDOW_AUTOSIZE);
    // cv::imshow(landmarkprint2, key2); //show image.
    // cv::waitKey();
  }


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
