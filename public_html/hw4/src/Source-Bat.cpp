/*
CS585 Image and Video Computing: Assignment 4

Author(s): Juan Santana, Ai Hue Nguyen
*/


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "math.h"
#include <stdlib.h>
#include <float.h>
#include <limits>
#include <time.h>


using namespace cv;
using namespace std;

//Structure Declarations 
struct centroid {
	Point coords;
	bool isBeingTracked;
};

//Class Declarations

// Hungarain Class libary came from...
// Source: https://github.com/mcximing/hungarian-algorithm-cpp/blob/master/Hungarian.h
// Hungarian Class functions came from...
// Source: https://github.com/mcximing/hungarian-algorithm-cpp/blob/master/Hungarian.cpp
class Hungarian
{
private:
	// Computes the optimal assignment (minimum overall costs)  
	void assignmentoptimal(int *assignment, double *cost, double *distMatrix, int nOfRows, int nOfColumns);
	void buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows, int nOfColumns);
	void computeassignmentcost(int *assignment, double *cost, double *distMatrix, int nOfRows);
	void step2a(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
	void step2b(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
	void step3(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
	void step4(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
	void step5(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
public:
	double Solve(vector<vector<double> >& DistMatrix, vector<int>& Assignment);
};

//Based off of OpenCV documents of Kalman Filter:
//Source: http://docs.opencv.org/master/dd/d6a/classcv_1_1KalmanFilter.html#gsc.tab=0, 10/17/2018
class TKalmanFilter
{
public:
	KalmanFilter* kalman;
	double deltatime;
	Point2f LastResult;
	TKalmanFilter(Point2f p, float dt = 0.2, float Accel_noise_mag = 0.5);
	~TKalmanFilter();
	Point2f GetPrediction();
	Point2f Update(Point2f p, bool DataCorrect);
};

// Track (object being tracked) Class Declaration 
class Track
{
public:
	vector<Point2d> trace;
	static size_t NextTrackID;
	size_t track_id;
	size_t skipped_frames;
	Point2d prediction;
	TKalmanFilter* KF;
	Track(Point2f p, float dt, float Accel_noise_mag);
};

// Tracker (tracking object) Class Declaration
class Tracker
{
public:

	// Filter Step time 
	float dt;

	float Accel_noise_mag;

	// Threshold distance
	double dist_threshold;

	// If points are separated by a distance > threshold, then this pair is NOT considered the problem of assignments

	// Max # of frames that track saves to not receive data measurements
	int maximum_allowed_skipped_frames;

	// Track's Max length
	int max_trace_length;

	vector<Track*> tracks;
	void Update(vector<Point2d>& detections);

	//Tracker & Other Parameters
	Tracker(float _dt, float _Accel_noise_mag, double _dist_threshold = 60, int _maximum_allowed_skipped_frames = 10, int _max_trace_length = 10);
	~Tracker(void);
};

// Function Declarations


/**
Reads the label maps contained in the text file and converts the label maps
into a labelled matrix, as well as a binary image containing the segmented objects

@param filename The filename string
@param labelled The matrix that will contain the object label information
Background is labelled 0, and object labels start at 1
@param binary The matrix that will contain the segmented objects that can be displayed
Background is labelled 0, and objects are labelled 255
*/
//From Lab 8
void convertFileToMat(String filename, Mat& labelled, Mat& binary);

/**
Reads the x-y object detections from the text file and draws in on a black and white image
@param filename The filename string
@param binary3channel The 3 channel image corresponding to the 1 channel binary image outputted
by convertFileToMat. The object detection positions will be drawn on this image
*/
// From Lab 8 
void drawObjectDetections(String filename, vector<centroid>& centroids);

int main()
{
	int start = 750; //Number in the first file's name 
	int final = 900;  // Number in the last file's name 
	int iter = final - start + 1;

	vector<centroid> centroids;

	Mat labelled, binary;
	Mat frame;

	Tracker tracker(0.2, 0.5, 60.0, 10, 10);

	//Different colors for paths of bats being tracked
	Scalar Colors[] = { Scalar(57,255,20), Scalar(70,102,255),Scalar(255, 7, 58), Scalar(188, 19, 254),Scalar(204, 255, 0),Scalar(255,103,0),Scalar(254,1,154),Scalar(127,0,255),Scalar(127,0,127), Scalar(150,75,0), Scalar(255,215,0), Scalar(255,218,185) };

	frame = Mat::zeros(Size(1024, 1024), CV_8UC3);

	//Loop to iterate through all 150 bat frames
	for (int i = 0; i < iter; i++) {

		//Clear centroids data from the pervious iteration
		centroids.clear();

		// Sets of operation to assemble the current segmentation file and localization file being looked at
		string segmentation_file, localization_file;
		string segmentation_start = "./Segmentation_Bats/CS585Bats-Segmentation_frame000000";
		string localization_start = "./Localization_Bats/CS585Bats-Localization_frame000000";
		string end = ".txt";
		string file_num = static_cast<ostringstream*>(&(ostringstream() << (start + i)))->str();
		if (file_num.size() == 1) file_num = "00" + file_num;
		else if (file_num.size() == 2) file_num = "0" + file_num;

		//Adding all strings together
		segmentation_file.append(segmentation_start); segmentation_file.append(file_num); segmentation_file.append(end);
		localization_file.append(localization_start); localization_file.append(file_num); localization_file.append(end);

		cout << "Bat Segmentation & Localization File: Frame " << file_num << endl;

		convertFileToMat(segmentation_file, labelled, binary);
		Mat binary3C;
		cvtColor(binary, binary3C, CV_GRAY2BGR);
		drawObjectDetections(localization_file, centroids);

        //Count up all the centroids (objects of interest) in the frame
		vector<Point2d> centers;
		for (int i = 0; i < centroids.size(); i++)
		{
			centers.push_back(centroids[i].coords);
		}

		if (centers.size() > 0)
		{
			tracker.Update(centers);

            // Display number of bats (centroids) detected
			cout << "Number of Bats Detected: " << tracker.tracks.size() << endl;

			add(frame, binary3C, frame);

			for (int i = 0; i < tracker.tracks.size(); i++)
			{
				if (tracker.tracks[i]->trace.size() > 1)
				{
					for (int j = 0; j < tracker.tracks[i]->trace.size() - 1; j++)
					{
						line(frame, tracker.tracks[i]->trace[j], tracker.tracks[i]->trace[j + 1], Colors[tracker.tracks[i]->track_id % 9], 2, CV_AA);
					}
				}
			}
		}

		//A "video" of the bat tracking frames put together one after another
		imshow("Slideshow of Bat Tracking Frames", frame);
		frame = frame - binary3C;
		waitKey(20);
	}

	char key = waitKey(0);
	return 0;
}

// From Lab 8
void convertFileToMat(String filename, Mat& labelled, Mat& binary)
{
	//Read file
	ifstream infile(filename.c_str());
	vector <vector <int> > data;
	if (!infile) {
		cout << "Reading File ERROR\t" << filename << "\n";
		return;
	}

	//Read the comma separated values into a vector of vector of ints
	while (infile)
	{
		string s;
		if (!getline(infile, s)) break;

		istringstream ss(s);
		vector <int> datarow;

		while (ss)
		{
			string srow;
			int sint;
			if (!getline(ss, srow, ',')) break;
			sint = atoi(srow.c_str()); //convert string to int
			datarow.push_back(sint);
		}
		data.push_back(datarow);
	}

	//Construct the labelled matrix from the vector of vector of ints
	labelled = Mat::zeros(data.size(), data.at(0).size(), CV_8UC1);
	for (int i = 0; i < labelled.rows; ++i)
		for (int j = 0; j < labelled.cols; ++j)
			labelled.at<uchar>(i, j) = data.at(i).at(j);

	//Construct the binary matrix from the labelled matrix
	binary = Mat::zeros(labelled.rows, labelled.cols, CV_8UC1);
	for (int i = 0; i < labelled.rows; ++i)
		for (int j = 0; j < labelled.cols; ++j)
			binary.at<uchar>(i, j) = (labelled.at<uchar>(i, j) == 0) ? 0 : 255;

}


void drawObjectDetections(String filename, vector<centroid>& centroids)
{
	ifstream infile(filename.c_str());
	vector <vector <int> > data;
	if (!infile) {
		cout << "Centroid file ERROR";
		return;
	}

	//read the comma separated values into a vector of vector of ints
	while (infile)
	{
		string s;
		if (!getline(infile, s)) break;

		istringstream ss(s);
		vector <int> datarow;

		while (ss)
		{
			string srow;
			int sint;
			if (!getline(ss, srow, ',')) break;
			sint = atoi(srow.c_str()); //Convert string to int
			datarow.push_back(sint);
		}
		data.push_back(datarow);
	}
    
	centroid cent;
	centroids.clear();
	for (int i = 0; i < data.size(); ++i)
	{
		Point center(data.at(i).at(0), data.at(i).at(1));
		cent.coords = center;
		cent.isBeingTracked = false;
		centroids.push_back(cent);
	}
}

//Hungarian Class Definition
//Function wrapper for solving assignment problem
double Hungarian::Solve(vector<vector<double> >& DistMatrix, vector<int>& Assignment)
{
	int max_x = DistMatrix[0].size();// x = rows = measurements
	int max_y = DistMatrix.size(); // y = columns = tracks


	int *assignment = new int[max_y];
	double *distIn = new double[max_y*max_x];

	double cost;
	// Fill matrix w/ random numbers
	for (int y = 0; y < max_y; y++)
	{
		for (int x = 0; x < max_x; x++)
		{
			distIn[y + max_y * x] = DistMatrix[y][x];
		}
	}
	assignmentoptimal(assignment, &cost, distIn, max_y, max_x);

	// Result
	Assignment.clear();
	for (int x = 0; x < max_y; x++)
	{
		Assignment.push_back(assignment[x]);
	}

	delete[] assignment;
	delete[] distIn;
	return cost;
}
// Computes optimal assignment / minimum overall costs using Munkres algorithm
void Hungarian::assignmentoptimal(int *assignment, double *cost, double *distMatrixIn, int nOfRows, int nOfColumns)
{
	double *distMatrix;
	double *distMatrixTemp;
	double *distMatrixEnd;
	double *columnEnd;
	double  value;
	double  minValue;

	bool *coveredColumns;
	bool *coveredRows;
	bool *starMatrix;
	bool *newStarMatrix;
	bool *primeMatrix;

	int nOfElements;
	int minDim;
	int row;
	int col;
	*cost = 0;
	for (row = 0; row < nOfRows; row++)
	{
		assignment[row] = -1.0;
	}

	// Generate distance matrix and check matrix elements positiveness

	// Total elements number
	nOfElements = nOfRows * nOfColumns;

	// Memory allocation
	distMatrix = (double *)malloc(nOfElements * sizeof(double));

	// Point to last element
	distMatrixEnd = distMatrix + nOfElements;

	for (row = 0; row < nOfElements; row++)
	{
		value = distMatrixIn[row];
		distMatrix[row] = value;
	}

	// Memory allocation
	coveredColumns = (bool *)calloc(nOfColumns, sizeof(bool));
	coveredRows = (bool *)calloc(nOfRows, sizeof(bool));
	starMatrix = (bool *)calloc(nOfElements, sizeof(bool));
	primeMatrix = (bool *)calloc(nOfElements, sizeof(bool));
	newStarMatrix = (bool *)calloc(nOfElements, sizeof(bool)); /* used in step4 */

	// Preliminary Steps
	if (nOfRows <= nOfColumns)
	{
		minDim = nOfRows;
		for (row = 0; row < nOfRows; row++)
		{
			// Find the smallest element in row 
			distMatrixTemp = distMatrix + row;
			minValue = *distMatrixTemp;
			distMatrixTemp += nOfRows;
			while (distMatrixTemp < distMatrixEnd)
			{
				value = *distMatrixTemp;
				if (value < minValue)
				{
					minValue = value;
				}
				distMatrixTemp += nOfRows;
			}
			// Subtract smallest element from each element of the row 
			distMatrixTemp = distMatrix + row;
			while (distMatrixTemp < distMatrixEnd)
			{
				*distMatrixTemp -= minValue;
				distMatrixTemp += nOfRows;
			}
		}

		// Steps 1 and Step 2a 
		for (row = 0; row < nOfRows; row++)
		{
			for (col = 0; col < nOfColumns; col++)
			{
				if (distMatrix[row + nOfRows * col] == 0)
				{
					if (!coveredColumns[col])
					{
						starMatrix[row + nOfRows * col] = true;
						coveredColumns[col] = true;
						break;
					}
				}
			}
		}
	}
	else // if(nOfRows > nOfColumns)
	{
		minDim = nOfColumns;
		for (col = 0; col < nOfColumns; col++)
		{
			// Find smallest element in column
			distMatrixTemp = distMatrix + nOfRows * col;
			columnEnd = distMatrixTemp + nOfRows;
			minValue = *distMatrixTemp++;
			while (distMatrixTemp < columnEnd)
			{
				value = *distMatrixTemp++;
				if (value < minValue)
				{
					minValue = value;
				}
			}
			// Subtract smallest element from each element of column 
			distMatrixTemp = distMatrix + nOfRows * col;
			while (distMatrixTemp < columnEnd)
			{
				*distMatrixTemp++ -= minValue;
			}
		}
		// Steps 1 & Step 2a 
		for (col = 0; col < nOfColumns; col++)
		{
			for (row = 0; row < nOfRows; row++)
			{
				if (distMatrix[row + nOfRows * col] == 0)
				{
					if (!coveredRows[row])
					{
						starMatrix[row + nOfRows * col] = true;
						coveredColumns[col] = true;
						coveredRows[row] = true;
						break;
					}
				}
			}
		}

		for (row = 0; row < nOfRows; row++)
		{
			coveredRows[row] = false;
		}
	}

	//Jump to Step 2b
	step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

	// Compute cost 
	// Remove invalid assignments
	computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);

	// Free up taken memory
	free(distMatrix);
	free(coveredColumns);
	free(coveredRows);
	free(starMatrix);
	free(primeMatrix);
	free(newStarMatrix);
	return;
}

// Hungarian Class Function buildassignmentvector
void Hungarian::buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows, int nOfColumns)
{
	int row, col;
	for (row = 0; row < nOfRows; row++)
	{
		for (col = 0; col < nOfColumns; col++)
		{
			if (starMatrix[row + nOfRows * col])
			{
				assignment[row] = col;
				break;
			}
		}
	}
}

// Hungarian Class Function computeassignmentcost
void Hungarian::computeassignmentcost(int *assignment, double *cost, double *distMatrix, int nOfRows)
{
	int row, col;
	for (row = 0; row < nOfRows; row++)
	{
		col = assignment[row];
		if (col >= 0)
		{
			*cost += distMatrix[row + nOfRows * col];
		}
	}
}

// Hungarian Class Function step2a
void Hungarian::step2a(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	bool *starMatrixTemp, *columnEnd;
	int col;

	// Cover every column containing a starred zero
	for (col = 0; col < nOfColumns; col++)
	{
		starMatrixTemp = starMatrix + nOfRows * col;
		columnEnd = starMatrixTemp + nOfRows;
		while (starMatrixTemp < columnEnd)
		{
			if (*starMatrixTemp++)
			{
				coveredColumns[col] = true;
				break;
			}
		}
	}
	// move to step 2b
	step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// Hungarian Class Function step2b
void Hungarian::step2b(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	int col, nOfCoveredColumns;
	// count covered columns 
	nOfCoveredColumns = 0;
	for (col = 0; col < nOfColumns; col++)
	{
		if (coveredColumns[col])
		{
			nOfCoveredColumns++;
		}
	}
	if (nOfCoveredColumns == minDim)
	{
		buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
	}
	else
	{
		// Move to step 3 
		step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
	}
}

// Hungarian Class Function step3
void Hungarian::step3(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	bool zerosFound;
	int row, col, starCol;
	zerosFound = true;
	while (zerosFound)
	{
		zerosFound = false;
		for (col = 0; col < nOfColumns; col++)
		{
			if (!coveredColumns[col])
			{
				for (row = 0; row < nOfRows; row++)
				{
					if ((!coveredRows[row]) && (distMatrix[row + nOfRows * col] == 0))
					{
						// Prime zero 
						primeMatrix[row + nOfRows * col] = true;
						// Find starred zero in current row 
						for (starCol = 0; starCol < nOfColumns; starCol++)
							if (starMatrix[row + nOfRows * starCol])
							{
								break;
							}
						if (starCol == nOfColumns) // no starred zero found 
						{
							// Move to step 4 
							step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
							return;
						}
						else
						{
							coveredRows[row] = true;
							coveredColumns[starCol] = false;
							zerosFound = true;
							break;
						}
					}
				}
			}
		}
	}
	// Move to step 5
	step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}


// Hungarian Class Function step4
void Hungarian::step4(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col)
{
	int n, starRow, starCol, primeRow, primeCol;
	int nOfElements = nOfRows * nOfColumns;

	// Generates temp copy of starMatrix 
	for (n = 0; n < nOfElements; n++)
	{
		newStarMatrix[n] = starMatrix[n];
	}
	// Star Current Zero
	newStarMatrix[row + nOfRows * col] = true;

	// Find Starred Zero in current col
	starCol = col;
	for (starRow = 0; starRow < nOfRows; starRow++)
	{
		if (starMatrix[starRow + nOfRows * starCol])
		{
			break;
		}
	}
	while (starRow < nOfRows)
	{
		// Unstar starred zero 
		newStarMatrix[starRow + nOfRows * starCol] = false;

		// Find primed zero in current row 
		primeRow = starRow;
		for (primeCol = 0; primeCol < nOfColumns; primeCol++)
		{
			if (primeMatrix[primeRow + nOfRows * primeCol])
			{
				break;
			}
		}
		// Star primed zero
		newStarMatrix[primeRow + nOfRows * primeCol] = true;

		// Find Starred Zero in current col
		starCol = primeCol;
		for (starRow = 0; starRow < nOfRows; starRow++)
		{
			if (starMatrix[starRow + nOfRows * starCol])
			{
				break;
			}
		}
	}

	// Use temp copy as new starMatrix
	// Delete all primes 
	// Uncover all rows
	for (n = 0; n < nOfElements; n++)
	{
		primeMatrix[n] = false;
		starMatrix[n] = newStarMatrix[n];
	}
	for (n = 0; n < nOfRows; n++)
	{
		coveredRows[n] = false;
	}

	// Jump to step 2a
	step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// Hungarian Class Function step5
void Hungarian::step5(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	double h, value;
	int row, col;

	// Find smallest uncovered element h
	h = DBL_MAX;
	for (row = 0; row < nOfRows; row++)
	{
		if (!coveredRows[row])
		{
			for (col = 0; col < nOfColumns; col++)
			{
				if (!coveredColumns[col])
				{
					value = distMatrix[row + nOfRows * col];
					if (value < h)
					{
						h = value;
					}
				}
			}
		}
	}
	// Add h to each covered row 
	for (row = 0; row < nOfRows; row++)
	{
		if (coveredRows[row])
		{
			for (col = 0; col < nOfColumns; col++)
			{
				distMatrix[row + nOfRows * col] += h;
			}
		}
	}
	// Subtract h from each uncovered column
	for (col = 0; col < nOfColumns; col++)
	{
		if (!coveredColumns[col])
		{
			for (row = 0; row < nOfRows; row++)
			{
				distMatrix[row + nOfRows * col] -= h;
			}
		}
	}
	// Jump to step 3
	step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// KalmanFilter Class Defintion 
TKalmanFilter::TKalmanFilter(Point2f pt, float dt, float Accel_noise_mag)
{
	// Time increment has to be small
	deltatime = dt;

	//4 state variables [x y v_x v_y]'
	//2 measurements [x y]
	kalman = new KalmanFilter(4, 2, 0);

	//Transition matrix, reflects usual position velocities
	kalman->transitionMatrix = (Mat_<float>(4, 4) << 1, 0, deltatime, 0, 0, 1, 0, deltatime, 0, 0, 1, 0, 0, 0, 0, 1);

	LastResult = pt;
	kalman->statePre.at<float>(0) = pt.x; // x
	kalman->statePre.at<float>(1) = pt.y; // y

	kalman->statePre.at<float>(2) = 0;
	kalman->statePre.at<float>(3) = 0;

	kalman->statePost.at<float>(0) = pt.x;
	kalman->statePost.at<float>(1) = pt.y;

	setIdentity(kalman->measurementMatrix);

	//Based off of http://answers.opencv.org/question/73190/how-to-improve-kalman-filter-performance/
	kalman->processNoiseCov = (Mat_<float>(4, 4) <<
		pow(deltatime, 4.0) / 4.0, 0, pow(deltatime, 3.0) / 2.0, 0,
		0, pow(deltatime, 4.0) / 4.0, 0, pow(deltatime, 3.0) / 2.0,
		pow(deltatime, 3.0) / 2.0, 0, pow(deltatime, 2.0), 0,
		0, pow(deltatime, 3.0) / 2.0, 0, pow(deltatime, 2.0));

	kalman->processNoiseCov *= Accel_noise_mag;

	setIdentity(kalman->measurementNoiseCov, Scalar::all(0.1));

	setIdentity(kalman->errorCovPost, Scalar::all(.1));

}

TKalmanFilter::~TKalmanFilter()
{
	// Delete to make more free memory
	delete kalman;
}

Point2f TKalmanFilter::GetPrediction()
{
	Mat prediction = kalman->predict();
	LastResult = Point2f(prediction.at<float>(0), prediction.at<float>(1));
	return LastResult;
}

Point2f TKalmanFilter::Update(Point2f p, bool DataCorrect)
{
	Mat measurement(2, 1, CV_32FC1);
	if (!DataCorrect) //Don't have measured position
	{
		//Update using prediction
		measurement.at<float>(0) = LastResult.x;
		measurement.at<float>(1) = LastResult.y;
	}
	else //Have measured position
	{
		//Update using measurements
		measurement.at<float>(0) = p.x;
		measurement.at<float>(1) = p.y;
	}
	// Correction
	Mat estimated = kalman->correct(measurement);

	//update using measurements
	LastResult.x = estimated.at<float>(0);
	LastResult.y = estimated.at<float>(1);
	return LastResult;
}

//TRACK Class Defintion
size_t Track::NextTrackID = 0;

//Track Constructor
Track::Track(Point2f pt, float dt, float Accel_noise_mag)
{
	track_id = NextTrackID;

	NextTrackID++;

	// Each track has Kalman filter for next prediction
	KF = new TKalmanFilter(pt, dt, Accel_noise_mag);

	//Storing point coordinates for next prediction
	prediction = pt;
	skipped_frames = 0;
}

//Tracker Constructor
Tracker::Tracker(float _dt, float _Accel_noise_mag, double _dist_threshold, int _maximum_allowed_skipped_frames, int _max_trace_length)
{
	dt = _dt;
	Accel_noise_mag = _Accel_noise_mag;
	dist_threshold = _dist_threshold;
	maximum_allowed_skipped_frames = _maximum_allowed_skipped_frames;
	max_trace_length = _max_trace_length;
}

//Free up memory
Tracker::~Tracker(void)
{
	for (int i = 0; i < tracks.size(); i++)
	{
		delete tracks[i];
	}
	tracks.clear();
}

void Tracker::Update(vector<Point2d>& detections)
{
	//If no tracking has happened yet, 
	// then every point starts its own track; happens when t=0
	if (tracks.size() == 0)
	{
		// If no tracks yet
		for (int i = 0; i < detections.size(); i++)
		{
			Track* tr = new Track(detections[i], dt, Accel_noise_mag);
			tracks.push_back(tr);
		}
	}

	//If the tracking is already happening

	//# of Tracks
	int N = tracks.size();
	//# of detectors
	int M = detections.size();

	// Matrix of distances from N-th track to M-th detector
	vector< vector<double> > Cost(N, vector<double>(M));

	// Destination
	vector<int> assignment;

	// Create cost matrix
	double dist;
	for (int i = 0; i < tracks.size(); i++)
	{

		for (int j = 0; j < detections.size(); j++)
		{
			Point2d diff = (tracks[i]->prediction - detections[j]); //We substract our prediction with our detection
			dist = sqrtf(diff.x*diff.x + diff.y*diff.y); //Find the magnitude of the
			Cost[i][j] = dist;
		}
	}

	//Apply Hungarian Algorithm for the assignment problem
	Hungarian hungarian;
	hungarian.Solve(Cost, assignment);


	//Clear up the assignments with pairs having long distance >dist_threshold

	// Unassigned tracks
	vector<int> not_assigned_tracks;

	for (int i = 0; i < assignment.size(); i++)
	{
		if (assignment[i] != -1)
		{
			if (Cost[i][assignment[i]] > dist_threshold)
			{
				assignment[i] = -1;
				// Mark the unassigned tracks and increase the counter of dropped frames,
				// when the number of dropped frames exceed the threshold value, the track is erased
				not_assigned_tracks.push_back(i);
			}
		}
		else
		{
			// If the track is not assigned to a detector, then increment the skipped frames
			tracks[i]->skipped_frames++;
		}

	}

	//After certain frames, if track doesn't get assigned to a detector, remove it
	for (int i = 0; i < tracks.size(); i++)
	{
		if (tracks[i]->skipped_frames > maximum_allowed_skipped_frames)
		{
			delete tracks[i];
			tracks.erase(tracks.begin() + i);
			assignment.erase(assignment.begin() + i);
			i--;
		}
	}

	//Check for unassigned detectors
	vector<int> not_assigned_detections;
	vector<int>::iterator it;
	for (int i = 0; i < detections.size(); i++)
	{
		it = find(assignment.begin(), assignment.end(), i);
		if (it == assignment.end())
		{
			not_assigned_detections.push_back(i);
		}
	}


	//These will be considered new tracks, so we initialize it
	if (not_assigned_detections.size() != 0)
	{
		for (int i = 0; i < not_assigned_detections.size(); i++)
		{
			Track* tr = new Track(detections[not_assigned_detections[i]], dt, Accel_noise_mag);
			tracks.push_back(tr);
		}
	}

	// Update status of filters
	for (int i = 0; i < assignment.size(); i++)
	{

		//Start by calculating prediction
		tracks[i]->KF->GetPrediction();

		if (assignment[i] != -1) // If assigment has an update 
		{
			tracks[i]->skipped_frames = 0;
			tracks[i]->prediction = tracks[i]->KF->Update(detections[assignment[i]], 1);
		}
		else	// If not, continue to predict
		{
			tracks[i]->prediction = tracks[i]->KF->Update(Point2f(0, 0), 0);
		}

		if (tracks[i]->trace.size() > max_trace_length)
		{
			tracks[i]->trace.erase(tracks[i]->trace.begin(), tracks[i]->trace.end() - max_trace_length);
		}

		tracks[i]->trace.push_back(tracks[i]->prediction);
		tracks[i]->KF->LastResult = tracks[i]->prediction;
	}

}
