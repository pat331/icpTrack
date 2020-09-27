#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <opencv2/opencv.hpp>


#include "../include/loadRadar.h"
using namespace std;
using namespace cv;

const double pi = 3.14159265358979323846;
const double encoderVariableConvert = 891.26743027717;

// Load the txt file into mat object
Mat ReadMatFromTxt(string filename, int rows,int cols) {
    float m;
    Mat out = Mat::zeros(rows, cols, CV_8UC1);//Matrix to store values

    ifstream fileStream(filename);
    int cnt = 0;//index starts from 0
    while (fileStream >> m)
    {
        int temprow = cnt / cols;
        int tempcol = cnt % cols;
				std::cout << "M "<< m << std::endl;
        out.at<float>(temprow, tempcol) = m;
        cnt++;
    }
    return out;
}

// Load path for radar scan png files.
vector<String> loadPathRadarScanPngFiles( string radaScanPngFilesFolder ) {
  vector<String> pathRadarScan; // std::string in opencv2.4, but cv::String in 3.0
  string path = radaScanPngFilesFolder;
  cv::glob(path,pathRadarScan,false);
  return pathRadarScan;
}
////////////////////////////////////////////////////////////////////////////////

vector<string> loadTimeStamp( string timeStampsFile ) {

	std::ifstream timeStampsFile1(timeStampsFile);
	std::ifstream timeStampsFile2(timeStampsFile);
	if (!timeStampsFile1) std::cerr << "Could not open timeStampsFile1!" << std::endl;
	if (!timeStampsFile2) std::cerr << "Could not open timeStampsFile2!" << std::endl;

	int numberOfLinesInFile = 0;

	// passaggio non necessario: fatto per mancanza di conoscenze in c++
	while(!timeStampsFile1.eof()){
    string timeStamps;
		int controlNumber; // 0 or 1 it is a check for the radar Image

    timeStampsFile1 >> timeStamps >> controlNumber; // extracts 2 floating point values seperated by whitespace

    numberOfLinesInFile +=1;
	}

	int dimensionVectorTimeStamps = numberOfLinesInFile-1;
	std::vector<string> timeStampsVector(dimensionVectorTimeStamps);
	int provaDimensione = timeStampsVector.size();
	std::cout << "dimensione timeStampsVector "<< provaDimensione << std::endl;
	std::cout << "numberOfLinesInFile "<< numberOfLinesInFile << std::endl;
	int counter = 0;

	if(counter < numberOfLinesInFile){
		std::cout << "entro in prova" << std::endl;

	}

	while(!timeStampsFile2.eof()){
		string timeStamps;
		int controlNumber; // 0 or 1 it is a check for the radar Image

		timeStampsFile2 >> timeStamps >> controlNumber; // extracts 2 floating point values seperated by whitespace

		if(counter < dimensionVectorTimeStamps){
			timeStampsVector[counter] = timeStamps;
			counter+=1;
		}
	}

	return timeStampsVector;

}

// Radar Image Decoder
////////////////////////////////////////////////////////////////////////////////

// Radar timeStamps Decoder
std::vector<uint64_t> radarTimeStampsDecoder(Mat radarImage){

	std::vector<uint64_t> timeStampsArrayRadarImage(400,0);
	uint64_t timeStamps;
	uint8_t timeStampBeforeCasting [8];

	int counterColumns;
	int counterRaws = 0;

	do {
		counterColumns = 0;

		for(counterColumns; counterColumns <= 7; counterColumns++){

			int prova = counterRaws;

			uint8_t value = radarImage.at<uchar>(Point(counterColumns, prova));
			// uint64_t value = radarImage.at<uchar>(Point(counterColumns, prova));
			timeStampBeforeCasting[counterColumns] = value;
		}

		timeStamps = (uint64_t)(*(uint64_t*) timeStampBeforeCasting);
		timeStampsArrayRadarImage[counterRaws] = timeStamps;
		counterRaws+=1;
	} while(counterRaws<400);

	return timeStampsArrayRadarImage;

}

// Radar Azimuth Decoder

std::vector<double> radarAzimuthsDecoder(Mat radarImage){

	std::vector<uint16_t> sweepCounterArrayRadarImage(400,0);
	std::vector<double> azimuthsArrayRadarImage(400,0);

	uint64_t sweepCounter;
	uint8_t sweepCounterBeforeCasting [2];

	const int encoderSize = 5600;
	int counterColumns;
	int counterRaws = 0;

	do {
		counterColumns = 8;

		for(counterColumns; counterColumns <= 9; counterColumns++){

			uint8_t value = radarImage.at<uchar>(Point(counterColumns, counterRaws));
			int counterColumnsVariable = counterColumns-8;
			sweepCounterBeforeCasting[counterColumnsVariable] = value;
		}

		sweepCounter = (uint64_t)(*(uint64_t*) sweepCounterBeforeCasting);
		sweepCounterArrayRadarImage[counterRaws] = sweepCounter;
		counterRaws+=1;
	} while(counterRaws<400);


	for(int k = 0; k<400;k++){
		azimuthsArrayRadarImage[k] = ((double) sweepCounterArrayRadarImage[k])/encoderVariableConvert;
	}

	return azimuthsArrayRadarImage;
}

// Radar Power Readings along each azimuth
std::vector<std::vector<double>> radarPowerReadings(Mat radarImage){
	// Create a vector containing n
	// vectors of size m.
	vector<vector<double>> radarPowerReadingsVector( 400 , vector<double> (3768));
	std::vector<uint64_t> timeStampsArrayRadarImage(400,0);

	int counterColumns;
	int counterRaws = 0;

	do {
		// Counter Columns = 12; from column 12 to the end are stored the power reading of the radar
		counterColumns = 12;

		for(counterColumns; counterColumns <= 3779; counterColumns++){


			uint64_t value = radarImage.at<uchar>(Point(counterColumns, counterRaws));
			double fft_data = (double)value/255;
			int convertitorCounterColumns = counterColumns-12;
			radarPowerReadingsVector[counterRaws][convertitorCounterColumns] = fft_data;
		}

		counterRaws+=1;

	} while(counterRaws<400);

	return radarPowerReadingsVector;

}
////////////////////////////////////////////////////////////////////////////////

// Radar to Polar radar image
void polarToCartesian(std::vector<double> azimuths, vector<vector<double>> radarPowerReadings){

	// Hard setup variable
	float radarResolution = 0.0432;
	float cartResolution = 0.25;
	int cartPixelWidth = 401;
	// Setup variable
	float cartMinRange;
	std::vector<float> coords;
	std::vector<std::vector<float>> Y;
	std::vector<std::vector<float>> X;
	std::vector<std::vector<float>> sampleRange;
	std::vector<std::vector<float>> sampleAngle;
	std::vector<float> sampleU;
	std::vector<float> sampleV;



	// vector<vector<float>> X;


	if (cartPixelWidth % 2 == 0){
		cartMinRange = (cartPixelWidth / 2 - 0.5) * cartResolution;
	}else{
		cartMinRange = floor(cartPixelWidth / 2 * cartResolution);
	}

	coords = linSpace(-cartMinRange, cartMinRange, cartPixelWidth);
	Y = meshGridY(coords);
	X = meshGridX(coords);
	// for(int f=0; f<20;f++){
	//
	// }
	// std::cout << "one dimension " << X.size() << std::endl;
	sampleRange = distanceOnGrid(X,Y);
	sampleAngle = angleOnGrid(X,Y);
	sampleAngle = convertAtanAngleResult(sampleAngle);

	// Interpolate Radar Data Coordinates
  double azimuthStep = azimuths[1] - azimuths[0];
	sampleU = calculateSampleU(sampleRange,radarResolution);
	sampleV = calculateSampleV(sampleAngle,azimuthStep,azimuths[0]);

// We clip the sample points to the minimum sensor reading range so that we
// do not have undefined results in the centre of the image. In practice
// this region is simply undefined.
	for(int k=0; k<sampleU.size();k++){
		if(sampleU[k]<0){
			sampleU[k] = 0;
		}
	}


	// for(int f=0; f<400;f++){
	// 	std::cout << "X " << X[0][f] << std::endl;
	// 	// std::cout << "Y " << Y[0][f] << std::endl;
	// }
	// std::cout << "size sampleV " << sampleV.size() << std::endl;
	// for(int f=0; f<400;f++){
	// 	std::cout << "sampleV " << sampleV[f] << std::endl;
	// }





}
////////////////////////////////////////////////////////////////////////////////
// Utility for cartesian to polar
// linSpace
std::vector<float> linSpace(float min, float max, int numberOfIntervals){
	int intervalLenght;
	float subIntervalLenght;

	if(max>0 && min>0){
		intervalLenght = max - min;
	}
	if(max>=0 && min <=0){
		intervalLenght = max + abs(min);
	}
	if(max <=0 && min <0){
		intervalLenght = abs(min) - abs(max);
	}
	subIntervalLenght = (float)intervalLenght/(float)(numberOfIntervals-1);
	std::vector<float> coords (numberOfIntervals,0);

	for (int i = 0; i < numberOfIntervals; i++){
		coords[i] = min + i*subIntervalLenght;
	}
	return coords;
}
// Meshgrid function
// MeshgriY
std::vector<std::vector<float>> meshGridY(std::vector<float> coords){
	int gridLenght;
	gridLenght = coords.size();
	vector<vector<float>> Y( gridLenght , vector<float> (gridLenght));

	for(int i = 0; i<gridLenght; i++){
		for(int j = 0; j<gridLenght; j++){
			Y[i][j] = coords[j];
		}
	}
	return Y;
}
// meshGridX
std::vector<std::vector<float>> meshGridX(std::vector<float> coords){
	int gridLenght;
	gridLenght = coords.size();
	vector<vector<float>> X( gridLenght , vector<float> (gridLenght));

	for(int i=0; i<gridLenght; i++){
		for(int j=0; j<gridLenght; j++){
			X[i][j] = coords[gridLenght-i-1];
		}
	}
	return X;
}
//
std::vector<std::vector<float>> distanceOnGrid(vector<vector<float>>  X, vector<vector<float>>  Y){

	int gridLenght = X.size();
	float squareDistance;
	std::vector<std::vector<float>> sampleRange(gridLenght, vector<float> (gridLenght));

	for(int i=0; i<gridLenght; i++){
		for(int j=0; j<gridLenght; j++){

			squareDistance = pow(X[i][j],2)+pow(Y[i][j],2);
			sampleRange[i][j] = sqrt(squareDistance);

		}
	}

	return sampleRange;
}

std::vector<std::vector<float>> angleOnGrid(vector<vector<float>>  X, vector<vector<float>>  Y){

	int gridLenght = X.size();
	std::vector<std::vector<float>> sampleAngle(gridLenght, vector<float> (gridLenght));

	for(int i=0; i<gridLenght; i++){
		for(int j=0; j<gridLenght; j++){
			// CONTROLLARE CON QUALI ELEMENTI DI X E Y CON CUI FAI ATAN2 ( ORDINE NEL FOR)
			sampleAngle[i][j] = atan2f(Y[i][j],X[i][j]);
			// sampleAngle[i][j] = atan2f(X[i][j],Y[i][j]);
		}
	}

	return sampleAngle;
}
std::vector<std::vector<float>> convertAtanAngleResult(vector<vector<float>>  sampleAngle){
	int gridLenght = sampleAngle.size();

	for(int i=0; i<gridLenght; i++){
		for(int j=0; j<gridLenght; j++){
			if(sampleAngle[i][j]<0){
				sampleAngle[i][j] += 2*pi;
			}
		}
	}
	return sampleAngle;

}

std::vector<float> calculateSampleU(std::vector<std::vector<float>> sampleRange, float radarResolution){
	int gridLenght = sampleRange.size();
	std::vector<float> sampleU (gridLenght*gridLenght,0);
	int k = 0;

	for(int i=0; i<gridLenght; i++){
		for(int j=0; j<gridLenght; j++){
				// sample_u = (sample_range - radar_resolution / 2) / radar_resolution
				sampleU[k] = (sampleRange[i][j] - radarResolution/2) /radarResolution;
				k++;
			}
		}
		return sampleU;
	}

	std::vector<float> calculateSampleV(std::vector<std::vector<float>> sampleAngle, double azimuthStep, double firstAzimuth){
		int gridLenght = sampleAngle.size();
		std::vector<float> sampleV (gridLenght*gridLenght,0);
		int k = 0;

		for(int i=0; i<gridLenght; i++){
			for(int j=0; j<gridLenght; j++){
					// sample_v = (sample_angle - azimuths[0]) / azimuth_step
					sampleV[k] = (sampleAngle[i][j] -firstAzimuth)/azimuthStep;
					k++;
				}
			}
			return sampleV;
		}


////////////////////////////////////////////////////////////////////////////////
