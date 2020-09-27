#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Load of txt file in mat object
Mat ReadMatFromTxt(string filename, int rows,int cols);
//
vector<String> loadPathRadarScanPngFiles( string radaScanPngFilesFolder );
vector<string> loadTimeStamp( string folderTimeStampsFile );

// Radar Image Decoder
std::vector<uint64_t> radarTimeStampsDecoder(Mat radarImage);
std::vector<double> radarAzimuthsDecoder(Mat radarImage);
std::vector<std::vector<double>> radarPowerReadings(Mat radarImage);
// Polar to Cartesian radar Imae radarPowerReadings
void polarToCartesian(std::vector<double> azimuths, vector<vector<double>> radarPowerReadings);
std::vector<float> linSpace(float min, float max, int numberOfIntervals);
std::vector<std::vector<float>> meshGridY(std::vector<float> coords);
std::vector<std::vector<float>> meshGridX(std::vector<float> coords);
std::vector<std::vector<float>> distanceOnGrid(vector<vector<float>>  X, vector<vector<float>>  Y);
std::vector<std::vector<float>> angleOnGrid(vector<vector<float>>  X, vector<vector<float>>  Y);
std::vector<std::vector<float>> convertAtanAngleResult(vector<vector<float>>  sampleAngle);
std::vector<float> calculateSampleU(std::vector<std::vector<float>> sampleRange, float radarResolution);
std::vector<float> calculateSampleV(std::vector<std::vector<float>> sampleAngle, double azimuthStep, double firstAzimuth);
