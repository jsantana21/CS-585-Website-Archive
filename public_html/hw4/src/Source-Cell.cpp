/*
CS585 Image and Video Computing
Assignment 4
Cell Segementation and Tracking Component
Aythor(s): Ai Hue Nguyen, Juan Santana
*/

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "math.h"

using namespace cv;
using namespace std;

struct tracked_cell {
	Point2d last;
	Point2d curr;
	Point2d projected;
	bool tracked;
	Scalar color;
};

struct centroid {
	Point2d coordinates;
	bool isTracked;
};

void greedyTracking(vector<tracked_cell>& cells, vector<centroid> &centroids);
void drawPath(vector<tracked_cell>& cells, Mat &canvas);
void alphaBetaFilter(tracked_cell *cell);
void segmentCells(Mat& img, Mat& binary_img);
void labelCells(Mat& img, Mat& labelled_img);

// Variables needed for alpha-beta filter 
double a = .85;
double b = .005;
double vx;
double vx1 = 0;
double projected_x_last = 0;
double projected_x;
double vy;
double vy1 = 0;
double projected_y_last = 0;
double projected_y;

double MIN_RADIUS = 20;

int main()
{
	// Example of cell segmentation on a particular frame 
	Mat cells_img = imread("CS585-Cells/t1265.tif", 1);
	Mat binary_cells_img, labelled_cells_img;
	segmentCells(cells_img, binary_cells_img);
	labelCells(binary_cells_img, labelled_cells_img);

	// Display results of cell segmentation 
	namedWindow("Cells", 1);
	imshow("Cells", cells_img);
	imshow("Binary Cells", binary_cells_img);
	labelCells(binary_cells_img, labelled_cells_img);
	imshow("Labeled Cells", labelled_cells_img);
	int initial = 265;
	int final = 375;
	int iter = final - initial + 1;

	// Cell tracking 
	// These vectors will be used to keep track of cells that have already been tracked and their centroids 
	vector<centroid> centroids;
	vector<tracked_cell> trackedCells;

	Scalar Colors[] = { Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),Scalar(255,255,0),Scalar(0,255,255),Scalar(255,0,255),Scalar(255,127,255),Scalar(127,0,255),Scalar(127,0,127) };
	Mat labelled, binary;
	Mat frame;
	Mat binary3C = Mat::zeros(Size(1070, 1036), CV_8UC3);

	// Loop over each frame in the dataset, creating a video sequence
	for (int i = 0; i < iter; i++) {
		centroids.clear();
		string filename;
		string ini_file = "CS585-Cells/t1";
		string ext = ".tif";
		string num1 = static_cast<ostringstream*>(&(ostringstream() << (initial + i)))->str();
		if (num1.size() == 1) num1 = "00" + num1;
		else if (num1.size() == 2) num1 = "0" + num1;
		filename.append(ini_file); filename.append(num1); filename.append(ext);
		Mat cells_img = imread(filename, 1);
		// Check for invalid input
		if (cells_img.empty())
		{
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}
		Mat binary_cells_img;
		segmentCells(cells_img, binary_cells_img);
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		// Find contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
		findContours(binary_cells_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		// Convert binary image back to color image
		Mat color_img;
		cvtColor(binary_cells_img, color_img, COLOR_GRAY2BGR);
		//draw red circles on the binary image
		for (int i = 0; i < contours.size(); ++i)
		{
			Moments shapeMoments = moments(contours[i]);
			float cx = shapeMoments.m10 / shapeMoments.m00;
			float cy = shapeMoments.m01 / shapeMoments.m00;

			Point center(cx, cy);
			circle(color_img, center, 3, Scalar(0, 0, 255), -1, 8);
		}
		// Compute centroid of each object 
		Moments shapeMoments = moments(contours[i]);
		float cx = shapeMoments.m10 / shapeMoments.m00;
		float cy = shapeMoments.m01 / shapeMoments.m00;
		Point center(cx, cy);
		centroid cent;
		cent.coordinates = center;
		cent.isTracked = false;
		centroids.push_back(cent);
		// Apply the alpha-beta filter 
		for (int k = 0; k < trackedCells.size(); k++)
		{
			alphaBetaFilter(&trackedCells[k]);
		}
		greedyTracking(trackedCells, centroids);
		// For every cell that has not already been tracked, create a new tracked_cell 
		int flag = 0;
		for (int j = 0; j < centroids.size(); j++) {
			// loop through centroids to find those that have not been previously tracked 
			if (!centroids.at(j).isTracked) {
				flag++;
				tracked_cell newCell;
				newCell.curr = centroids.at(j).coordinates;
				newCell.last = centroids.at(j).coordinates;
				newCell.projected = centroids.at(j).coordinates;
				int blue = rand() % 254;
				int green = rand() % 254;
				int red = rand() % 254;
				newCell.color = Colors[j % 9];
				trackedCells.push_back(newCell);
			}
		}
		drawPath(trackedCells, binary3C);
		add(color_img, binary3C, color_img);
		imshow("Cell Trajectories", color_img);
		waitKey(20);
		cout << "There are " << flag << " new cell(s) detected in in frame number " << i << endl;
	}
	char key = waitKey(0);
	return 0;
}

void drawPath(vector<tracked_cell>& cells, Mat &canvas)
{
	int thickness = 2;
	int lineType = 8;
	for (int i = 0; i < cells.size(); i++)
	{
		line(canvas, cells[i].last, cells[i].curr, cells[i].color, thickness, lineType);
	}
}

double findDist(int x1, int y1, int x2, int y2) {
	return sqrt(pow(abs(x2 - x1), 2) + pow(abs(y2 - y1), 2));
}

// computes distance between objects in subsequent frames and makes sure that objects that have been tracked are marked as tracked 
void greedyTracking(vector<tracked_cell>& cells, vector<centroid>& centroids) {
	int maxX = 1070;
	int maxY = 1036;

	for (int i = 0; i < cells.size(); i++) {
		int projX = cells.at(i).projected.x;
		int projY = cells.at(i).projected.y;
		double minDif = 999999;
		int centroidIndex = -1;
		if (projX > maxX || projY > maxY || projX < 0 || projY < 0) {
			cells.erase(cells.begin() + i);
		}
		else {
			for (int j = 0; j < centroids.size(); j++) {
				if (!centroids.at(j).isTracked) {
					double distance = findDist(projX, projY, centroids.at(j).coordinates.x, centroids.at(j).coordinates.y);
					if (distance < minDif) {
						minDif = distance;
						centroidIndex = j;
					}
				}
			}
			if (minDif > MIN_RADIUS) {
				for (int j = 0; j < centroids.size(); j++) {
					if (centroids.at(j).isTracked) {
						double distance = findDist(projX, projY, centroids.at(j).coordinates.x, centroids.at(j).coordinates.y);
						if (distance < minDif) {
							minDif = distance;
							centroidIndex = j;
						}
					}
				}
			}

			if (centroidIndex == -1) {
				cells.erase(cells.begin() + i);
			}

			else {
				cells.at(i).last = cells.at(i).curr;
				cells.at(i).curr = centroids.at(centroidIndex).coordinates;
				centroids.at(centroidIndex).isTracked = true;

			}
		}
	}
}

// Based on code from here: https://en.wikipedia.org/wiki/Alpha_beta_filter
void alphaBetaFilter(tracked_cell *cell) {
	projected_x = projected_x_last + vx1;
	vx = vx1;
	projected_y = projected_y_last + vy1;
	vy = vy1;

	// update residuals and projections 
	double rx = cell->curr.x - projected_x;
	double ry = cell->curr.y - projected_y;
	projected_x += a * rx;
	vx += (b * rx);
	projected_y += a * ry;
	vy += (b * ry);

	projected_x_last = projected_x;
	vx1 = vx;
	projected_y_last = projected_y;
	vy1 = vy;

	// update projections for x and y 
	cell->projected.x = projected_x;
	cell->projected.y = projected_y;
}

void segmentCells(Mat& img, Mat& binary_img) {
	// Convert input image to grayscale
	Mat dest = Mat::zeros(img.rows, img.cols, CV_8UC3);
	cvtColor(img, dest, CV_BGR2GRAY);

	// Convert grayscale image into binary image using thresholding (based on code from Lab5) 
	cv::threshold(dest, binary_img, 150, 255, 0);
}

void labelCells(Mat& img, Mat& labelled_img) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	// Find contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
	findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Convert input image back into a color image so that circles in next part will show up as red 
	cvtColor(img, labelled_img, COLOR_GRAY2BGR);

	// draw red circles on the binary image
	for (int i = 0; i < contours.size(); ++i)
	{
		Moments shapeMoments = moments(contours[i]);
		float cx = shapeMoments.m10 / shapeMoments.m00;
		float cy = shapeMoments.m01 / shapeMoments.m00;
		Point center(cx, cy);
		circle(labelled_img, center, 3, Scalar(0, 0, 255), -1, 8);
	}
}