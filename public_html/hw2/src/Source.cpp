//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//C++ standard libraries
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

//function declarations
vector<Point> MatchingMethod(Mat &img_bw, Mat &img_color, Mat &templ, string name);
void mySkinDetect(Mat& src, Mat& dst);
int myMax(int a, int b, int c);
int myMaxVal(double* array, int size);
int myMin(int a, int b, int c);
vector<Point> myContour(Mat &src, vector < vector< Point > > &contours, string name, int &largest, vector<Vec4i> &hierarchy);



int rmin = 45, rmax = 192, bmin = 18, bmax = 147, gmin = 76, gmax = 151;

double match[3];


int main(int argc, char** argv)
{

	// Reading the camera
	VideoCapture cap(0);

	// if not successful, exit program
	if (!cap.isOpened())
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	//Images for the templates
	Mat template1, template2, template3, template4;


	//Read the different templates
	template1 = imread("okay.jpg", CV_32FC1);
	template2 = imread("rocknroll.png", CV_32FC1);
	template3 = imread("peacesign.png", CV_32FC1);
	template4 = imread("thumbsupsign.jpg", CV_32FC1);

	Mat template1_output, template2_output, template3_output, template4_output; //Images for skinDetection templates 
	vector<vector<Point> > cont1, cont2, cont3, cont4; //Vector of contours
	vector<Point> max_cont1, max_cont2, max_cont3, max_cont4; //Maximum contourto define shape of the hand

	//Initialize skinDetection templates
	template1_output = Mat::zeros(template1.rows, template1.cols, CV_8UC1);
	template2_output = Mat::zeros(template2.rows, template2.cols, CV_8UC1);
	template3_output = Mat::zeros(template3.rows, template3.cols, CV_8UC1);
	template4_output = Mat::zeros(template4.rows, template4.cols, CV_8UC1);

	//Perform skinDetection
	mySkinDetect(template1, template1_output);
	mySkinDetect(template2, template2_output);
	mySkinDetect(template3, template3_output);
	mySkinDetect(template4, template4_output);

	vector<Vec4i> hier1, hier2, hier3, hier4; //to be used in the findContour function

	// Based of code from lab2 & lab3


	//Finding contour for template 1
	findContours(template1_output, cont1, hier1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//Find the largest contour
	int idx1 = 0;
	double max_area1 = 0; int max_contour_idx1;
	for (; idx1 < cont1.size(); idx1++)
	{
		double area1 = contourArea(cont1[idx1]);
		if (area1 > max_area1)
		{
			max_area1 = area1;
			max_contour_idx1 = idx1;
		}
	}
	max_cont1 = cont1[max_contour_idx1];
	//Draw  object with the largest contour and its outline on a new image
	Scalar color1 = Scalar(0, 255, 0);
	Mat contourtemplate1 = Mat::zeros(template1_output.size(), CV_8UC3);
	drawContours(contourtemplate1, cont1, max_contour_idx1, color1, 3, 8, hier1, 0, Point());
	imshow("Okay Sign Template", contourtemplate1);

	//Finding contour for template 2
	findContours(template2_output, cont2, hier2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//Find the largest contour
	int idx2 = 0;
	double max_area2 = 0; int max_contour_idx2;
	for (; idx2 < cont2.size(); idx2++)
	{
		double area2 = contourArea(cont2[idx2]);
		if (area2 > max_area2)
		{
			max_area2 = area2;
			max_contour_idx2 = idx2;
		}
	}
	max_cont2 = cont2[max_contour_idx2];
	//Draw the object with the largest contour and its outline on a new image
	Scalar color2 = Scalar(0, 255, 0);
	Mat contourtemplate2 = Mat::zeros(template2_output.size(), CV_8UC3);
	drawContours(contourtemplate2, cont2, max_contour_idx2, color2, 3, 8, hier2, 0, Point());
	imshow("Rock n' Roll Template ", contourtemplate2); 

	//Finding contour for template 3
	findContours(template3_output, cont3, hier3, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//Find the largest contour
	int idx3 = 0;
	double max_area3 = 0; 
	int max_contour_idx3;
	for (; idx3 < cont3.size(); idx3++)
	{
		double area3 = contourArea(cont3[idx3]);
		if (area3 > max_area3)
		{
			max_area3 = area3;
			max_contour_idx3 = idx3;
		}
	}
	max_cont3 = cont3[max_contour_idx3];
	//Draw the object with the largest contour and its outline on a new image
	Scalar color3 = Scalar(0, 255, 0);
	Mat contourtemplate3 = Mat::zeros(template3_output.size(), CV_8UC3);
	drawContours(contourtemplate3, cont3, max_contour_idx3, color3, 3, 8, hier3, 0, Point());
	imshow("Peace Sign Template", contourtemplate3);

	//Finding contour for template 4
	findContours(template4_output, cont4, hier4, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//Find the largest contour
	int idx4 = 0;
	double max_area4 = 0;
	int max_contour_idx4;
	for (; idx4 < cont4.size(); idx4++)
	{
		double area4 = contourArea(cont4[idx4]);
		if (area4 > max_area4)
		{
			max_area4 = area4;
			max_contour_idx4 = idx4;
		}
	}
	max_cont4 = cont4[max_contour_idx4];
	//Draw the object with the largest contour as well as its outline on a new image
	Scalar color4 = Scalar(0, 255, 0);
	Mat contourtemplate4 = Mat::zeros(template4_output.size(), CV_8UC3);
	drawContours(contourtemplate4, cont4, max_contour_idx4, color4, 3, 8, hier4, 0, Point());
	imshow("Thumbs Up Template", contourtemplate4); 

	//Create 4 vector<Point> that will store the contour for the camera using matching template w the 4 templates
	vector<Point> webcam_cont1, webcam_cont2, webcam_cont3, webcam_cont4;
	while (1)
	{
		// read a new frame from video
		Mat frame;
		bool bSuccess = cap.read(frame);

		//if not successful, break loop
		if (!bSuccess)
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		Mat frame_output;
		frame_output = Mat::zeros(frame.rows, frame.cols, CV_8UC1); 

		//Skin Detection of the webcam frame
		mySkinDetect(frame, frame_output);
		imshow("MyVideo", frame_output);

		//Perform matching method through the shapes
		webcam_cont1 = MatchingMethod(frame_output, frame, template1_output, "Okay");
		webcam_cont2 = MatchingMethod(frame_output, frame, template2_output, "Rock");
		webcam_cont3 = MatchingMethod(frame_output, frame, template3_output, "Peace");
		webcam_cont4 = MatchingMethod(frame_output, frame, template4_output, "Thumbs Up");

		//Calculate how similar each correspondent template contour and webcamera contour are
		match[0] = matchShapes(webcam_cont1, max_cont1, 1, 1);
		match[1] = matchShapes(webcam_cont2, max_cont2, 1, 1);
		match[2] = matchShapes(webcam_cont3, max_cont3, 1, 1);
		match[3] = matchShapes(webcam_cont4, max_cont4, 1, 1);

		//Find the maximum value of match array
		int max_value;
		max_value = myMaxVal(match, 4);

		//Output of the program: Visualize the values of the matchShapes and output what shape is begin recognized by the algorithm 
		cout << "Coefficients: " << match[0] << " , " << match[1] << " , " << match[2] << " , " << match[3] << endl;
		if (max_value == 0) cout << "OKAY" << endl;
		if (max_value == 1) cout << "ROCK " << endl;
		if (max_value == 2) cout << "PEACE" << endl;
		if (max_value == 3) cout << "THUMBS UP" << endl;

		//wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		if (waitKey(30) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
	cap.release();
	return 0;
}



vector<Point> MatchingMethod(Mat &img_bw, Mat &img_color, Mat &templ, string name)
{
	/// Source image to display
	Mat result;
	Mat img_display;
	img_color.copyTo(img_display);
	int match_method = CV_TM_CCORR_NORMED;
	/// Create the result matrix
	int result_cols = img_bw.cols - templ.cols + 1;
	int result_rows = img_bw.rows - templ.rows + 1;

	result.create(result_cols, result_rows, CV_32FC1);

	/// Do the Matching and Normalize
	matchTemplate(img_bw, templ, result, match_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	/// Show me what you got
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);

	Rect r(matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows));
	Mat ROI = img_bw(r);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(ROI, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//		4. Find the largest contour
	int idx = 0;
	double max_area = 0; int max_contour_idx;
	for (; idx < contours.size(); idx++)
	{
		double area = contourArea(contours[idx]);
		if (area > max_area)
		{
			max_area = area;
			max_contour_idx = idx;
		}
	}
	// Draw the object with the largest contour as well as its outline on a new image
	Scalar color = Scalar(0, 0, 255);
	Mat drawing = Mat::zeros(ROI.size(), CV_8UC3);
	drawContours(drawing, contours, max_contour_idx, color, 3, 8, hierarchy, 0, Point());
	imshow(name, drawing);

	return contours[max_contour_idx];
}

int myMax(int a, int b, int c) {
	int m = a;
	(void)((m < b) && (m = b));
	(void)((m < c) && (m = c));
	return m;
}

///Calculates the maximum index of the array
int myMaxVal(double* array, int size)
{
	double max = array[0];
	int max_pos = 0;
	int i;
	for (i = 0; i < size; i++)
	{
		if (max < array[i])
		{
			max = array[i];
			max_pos = i;
		}
	}
	return max_pos;
}

int myMin(int a, int b, int c) {
	int m = a;
	(void)((m > b) && (m = b));
	(void)((m > c) && (m = c));
	return m;
}

//Function that detects whether a pixel belongs to the skin based on RGB values
void mySkinDetect(Mat& src, Mat& dst) {
	//Surveys of skin color modeling and detection techniques:
	//Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
	//Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.

	inRange(src, Scalar(bmin, gmin, rmin), Scalar(bmax, gmax, rmax), dst);

}

///Returns  maximum contour
vector<Point> myContour(Mat &src, vector < vector< Point > > &contours, string name, int &largest, vector<Vec4i> &hierarchy)
{
	double area;
	double largest_area = 0;
	largest = 0;
	findContours(src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	Mat drawing = Mat::zeros(src.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++) {
		area = contourArea(contours[i], false);
		if (area > largest_area) {
			largest = i;
			largest_area = area;
		}
	}

	return contours[largest];
}