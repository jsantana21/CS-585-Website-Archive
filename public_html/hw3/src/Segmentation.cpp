/**
CS585 Image and Video Computing
HW 3: Object Shape Analysis and Segmentation Programming Assignment
--------------
Problem 2: Segmentation
--------------
Your data for this part are three color video sequences
--------------
Tasks:
1. Implement two segmentation algorithms (e.g., absolute or adaptive thresholding, region growing, etc.). Apply the algorithms to convert your color image sequence input into a binary video output.
Segmentation can be very challenging. It is acceptable, for this homework, to make some decisions to restrict the problem you are trying to solve, for example, by defining a region of interest (a "mask") and ignoring portions of the image outside of this region.
2. Apply the tools you implemented for part 1 to analyze your data.
--------------
*/

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

#include "opencv2/objdetect/objdetect.hpp"

using namespace cv;
using namespace std;

// Function headers
void threshold_callback1(int, void*);
void threshold_callback2(int, void*);

struct callback_data {
	Mat src_gray;
	int thresh;
	int max_thresh;
};

// Based on code in StackOverflow answer here: https://stackoverflow.com/questions/35668074/how-i-can-take-the-average-of-100-image-using-opencv
Mat getMean(const vector<Mat>& images)
{
	if (images.empty()) return Mat();

	// Create a 0 initialized image to use as accumulator
	Mat m(images[0].rows, images[0].cols, CV_64FC3);
	m.setTo(Scalar(0, 0, 0, 0));

	// Use a temp image to hold the conversion of each input image to CV_64FC3
	// This will be allocated just the first time, since all your images have
	// the same size.
	Mat temp;
	for (int i = 0; i < images.size(); ++i)
	{
		// Convert the input images to CV_64FC3 ...
		images[i].convertTo(temp, CV_64FC3);

		// ... so you can accumulate
		m += temp;
	}

	// Convert back to CV_8UC3 type, applying the division to get the actual mean
	m.convertTo(m, CV_8U, 1. / images.size());
	return m;
}

// Function that returns the maximum of 3 integers (from Lab 3) 
int myMax(int a, int b, int c) {
	int m = a;
	(void)((m < b) && (m = b)); //short-circuit evaluation
	(void)((m < c) && (m = c));
	return m;
}

// Function that returns the minimum of 3 integers (from Lab 3) 
int myMin(int a, int b, int c) {
	int m = a;
	(void)((m > b) && (m = b));
	(void)((m > c) && (m = c));
	return m;
}

// Function that detects whether a pixel belongs to the skin based on RGB values (from Lab 3) 
void mySkinDetect(Mat& src, Mat& dst) {
	//Surveys of skin color modeling and detection techniques:
	//Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
	//Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			//For each pixel, compute the average intensity of the 3 color channels
			Vec3b intensity = src.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)
			int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
			if ((R > 95 && G > 40 && B > 20) && (myMax(R, G, B) - myMin(R, G, B) > 15) && (abs(R - G) > 15) && (R > G) && (R > B)) {
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
}

// Function to detect number of people in each frame in the third data set 
// Based on code in question and answer here: https://www.dreamincode.net/forums/topic/301087-error-in-hog-descriptor-in-opencv/
void myPeopleDetect() {
	// frame in which the detection is being performed
	Mat frame = imread("CS585-PeopleImages/frame_0010.jpg");
	vector<Rect> found;
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	hog.detectMultiScale(frame, found, 0, Size(8, 8), Size(24, 16), 1.05, 2);
	for (int i = 0; i < (int)found.size(); i++) {
		Rect r = found[i];
		r.x += cvRound(r.width*0.1);
		r.y += cvRound(r.height*0.1);
		r.width = cvRound(r.width*0.8);
		r.height = cvRound(r.height*0.8);
		rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 1);
	}
	namedWindow("People Detector", 1);
	imshow("People Detector", frame);
	cout << "People Detected: " << (int)found.size() << endl;
}

void myHandsDetect() {
	vector<Mat> images;
	Mat img1 = imread("piano_14.png", 1);
	images.push_back(img1);
	Mat img2 = imread("piano_15.png", 1);
	images.push_back(img2);
	Mat img3 = imread("piano_16.png", 1);
	images.push_back(img3);
	Mat img4 = imread("piano_17.png", 1);
	images.push_back(img4);

	// Compute the average of all of the frames 
	Mat meanImage = getMean(images);
	imshow("Mean image", meanImage);

	// Compute the difference between a frame and the average 
	Mat frame = img1;
	cv::Mat diffImage;
	cv::absdiff(meanImage, frame, diffImage);
	cv::Mat foregroundMask = cv::Mat::zeros(diffImage.rows, diffImage.cols, CV_8UC1);
	float threshold = 30.0f;
	float dist;
	for (int j = 0; j < diffImage.rows; ++j)
		for (int i = 0; i < diffImage.cols; ++i)
		{
			cv::Vec3b pix = diffImage.at<cv::Vec3b>(j, i);
			dist = (pix[0] * pix[0] + pix[1] * pix[1] + pix[2] * pix[2]);
			dist = sqrt(dist);
			if (dist > threshold)
			{
				foregroundMask.at<unsigned char>(j, i) = 255;
			}
		}
	imshow("Difference Image", diffImage);

	// Convert difference image to gray
	Mat dest = Mat::zeros(diffImage.rows, diffImage.cols, CV_8UC1);
	cvtColor(diffImage, dest, CV_BGR2GRAY);
	imshow("Gray Image", dest);

	// Convert into binary image using thresholding (based on code from Lab5) 
	Mat output = Mat::zeros(diffImage.rows, diffImage.cols, CV_8UC1);
	cv::threshold(dest, output, 50, 255, 0);
	namedWindow("Threshold", CV_WINDOW_AUTOSIZE);
	imshow("Binary Image", output);

	// Dilate the binarized image to get a mask 
	Mat kernel = Mat::ones(3, 3, CV_32F);
	for (int i = 0; i < 15; i++) {
		morphologyEx(output, output, cv::MORPH_DILATE, kernel);
	}
	imshow("Mask", output);

	// Apply the mask 
	Mat res = Mat::zeros(output.rows, output.cols, CV_8UC1);
	bitwise_and(img1, img1, res, output);
	imshow("Frame after Masking", res);

	Mat frameDest = Mat::zeros(res.rows, res.cols, CV_8UC1);
	mySkinDetect(res, frameDest);
	imshow("Skin Detection Result", frameDest);

	
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	// Find contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
	findContours(frameDest, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Based on code in StackOverflow answer here: https://stackoverflow.com/questions/15012073/opencv-draw-draw-contours-of-2-largest-objects
	int largestIndex = 0;
	int largestContour = 0;
	int secondLargestIndex = 0;
	int thirdLargestIndex = 0;
	int secondLargestContour = 0;
	int thirdLargestContour = 0;
	Mat drawing1 = Mat::zeros(frameDest.size(), CV_8UC3);
	Mat drawing2 = Mat::zeros(frameDest.size(), CV_8UC3);
	Mat drawing3 = Mat::zeros(frameDest.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > largestContour) {
			secondLargestContour = largestContour;
			secondLargestIndex = largestIndex;
			largestContour = contours[i].size();
			largestIndex = i;
		}
		else if (contours[i].size() > secondLargestContour) {
			thirdLargestContour = secondLargestContour;
			thirdLargestIndex = secondLargestIndex;
			secondLargestContour = contours[i].size();
			secondLargestIndex = i;
		}
		else if (contours[i].size() > thirdLargestContour) {
			thirdLargestContour = contours[i].size();
			thirdLargestIndex = i;
		}
	}
	Scalar color = Scalar(0, 0, 255);
	drawContours(drawing1, contours, largestIndex, color, CV_FILLED, 8);
	drawContours(drawing2, contours, secondLargestIndex, color, CV_FILLED, 8);
	imshow("Largest Contour", drawing1);
	imshow("Second Largest Contour", drawing2);
	imshow("Third Largest Contour", drawing3);
}

void myBatDetect() {
	// Based on code here: https://riptutorial.com/opencv/example/22518/circular-blob-detection
	Mat img = imread("colored_bats.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat resultImg = Mat::zeros(img.rows, img.cols, CV_8UC1);
	cvtColor(img, resultImg, CV_GRAY2BGR);

	// threshold the image with gray value of 100
	Mat binImg = Mat::zeros(img.rows, img.cols, CV_8UC1);
	threshold(img, binImg, 100, 255, THRESH_BINARY);

	// find the contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binImg, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	if (contours.size() <= 0)
	{
		printf("no contours found");
	}

	drawContours(binImg, contours, (0, 0, 255), CV_FILLED, 8);
	imshow("Contours", binImg);
	
	/*
	// filter the contours
	vector<vector<Point>> filteredBlobs;
	Mat centers = Mat::zeros(0, 2, CV_64FC1);
	for (int i = 0; i < contours.size(); i++)
	{
		// calculate circularity
		double area = contourArea(contours[i]);
		double arclength = arcLength(contours[i], true);
		double circularity = 4 * CV_PI * area / (arclength * arclength);
		if (circularity > 0.8)
		{
			filteredBlobs.push_back(contours[i]);

			//calculate center
			Moments mu = moments(contours[i], false);
			Mat centerpoint = Mat(1, 2, CV_64FC1);
			centerpoint.at<double>(i, 0) = mu.m10 / mu.m00; // x-coordinate
			centerpoint.at<double>(i, 1) = mu.m01 / mu.m00; // y-coordinate
			centers.push_back(centerpoint);
		}
	}

	if (filteredBlobs.size() <= 0)
	{
		printf("no circular blobs found");
	}
	
	drawContours(resultImg, filteredBlobs, -1, Scalar(0, 0, 255), CV_FILLED, 8);

	imshow("Blobs", resultImg);
	*/
}

// main function
int main()
{
	myHandsDetect();
	myBatDetect();
	myPeopleDetect();

	waitKey();

	callback_data cbd;
	cbd.thresh = 128;
	cbd.max_thresh = 255;

	Mat src;

	// Load source image and convert it to gray
	src = imread("piano_14.png", 1);

	// Create Window and display source image
	namedWindow("Source", CV_WINDOW_AUTOSIZE);
	imshow("Source", src);

	// Convert image to gray
	cvtColor(src, cbd.src_gray, CV_BGR2GRAY);
	// Blur the image
	blur(cbd.src_gray, cbd.src_gray, Size(3, 3));

	// Create Trackbar
	// Documentation for createTrackbar: http://docs.opencv.org/modules/highgui/doc/user_interface.html?highlight=createtrackbar#createtrackbar
	// Example of adding a trackbar: http://docs.opencv.org/doc/tutorials/highgui/trackbar/trackbar.html

	// Method 1: Adaptive thresholding
	createTrackbar("Threshold1:", "Source", &cbd.thresh, cbd.max_thresh, threshold_callback1, &cbd);
	threshold_callback1(0, &cbd);

	// Method 2: Edge thresholding
	createTrackbar("Threshold2:", "Source", &cbd.thresh, cbd.max_thresh, threshold_callback2, &cbd);
	threshold_callback2(0, &cbd);

	// Wait until keypress
	waitKey(0);
	return(0);
}

// function threshold_callback 
void threshold_callback1(int, void* x)
{
	callback_data cbd = *(callback_data *)x;
	Mat thres_output;
	Mat a_thres_output;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	// Convert into binary image using thresholding
	// Documentation for threshold: http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#threshold
	// Example of thresholding: http://docs.opencv.org/doc/tutorials/imgproc/threshold/threshold.html
	threshold(cbd.src_gray, thres_output, cbd.thresh, cbd.max_thresh, 0);
	namedWindow("Threshold", CV_WINDOW_AUTOSIZE);
	imshow("Threshold", thres_output);

	// Documentation for adaptive thresholding: http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=adaptivethreshold#cv2.adaptiveThreshold
	adaptiveThreshold(cbd.src_gray, a_thres_output, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 17, -5);
	namedWindow("Adaptive threshold", CV_WINDOW_AUTOSIZE);
	imshow("Adaptive threshold", a_thres_output);

	// Find contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
	findContours(a_thres_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	Mat contour_output = Mat::zeros(thres_output.size(), CV_8UC3);
	cout << "The number of contours detected is: " << contours.size() << endl;


	//: Find largest contour
	int maxsize = 0;
	int maxind = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		// Documentation on contourArea: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#
		double area = contourArea(contours[i]);
		if (area > maxsize) {
			maxsize = area;
			maxind = i;
		}
	}
	cout << "The area of the largest contour detected is: " << contourArea(contours[maxind]) << endl;
	cout << "-----------------------------" << endl << endl;

	// Draw contours
	// Documentation for drawing contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours#drawcontours
	// drawContours(contour_output, contours, maxind, Scalar(255, 0, 0), CV_FILLED, 8, hierarchy);
	drawContours(contour_output, contours, maxind, Scalar(0, 0, 255), 2, 8, hierarchy);

	// Show in a window
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", contour_output);
}

// function threshold_callback 
void threshold_callback2(int, void* x)
{
	callback_data cbd = *(callback_data *)x;
	Mat thres_output;
	Mat a_thres_output;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	// Parameters
	int ksize = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	// Gradient results
	Mat grad, grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	// Gradient X
	Sobel(cbd.src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	// Gradient Y
	Sobel(cbd.src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	// Total Gradient (Approximately)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	namedWindow("Edge", CV_WINDOW_AUTOSIZE);
	imshow("Edge", grad);

	// Thresholding
	threshold(grad, grad, cbd.thresh, cbd.max_thresh, 0);
	namedWindow("Edge threshold", CV_WINDOW_AUTOSIZE);
	imshow("Edge threshold", grad);


	// Find contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
	findContours(grad, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	Mat contour_output = Mat::zeros(grad.size(), CV_8UC3);
	cout << "The number of contours detected is: " << contours.size() << endl;

	//: Find largest contour
	int maxsize = 0;
	int maxind = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		// Documentation on contourArea: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#
		double area = contourArea(contours[i]);
		if (area > maxsize) {
			maxsize = area;
			maxind = i;
		}
	}
	cout << "The area of the largest contour detected is: " << contourArea(contours[maxind]) << endl;
	cout << "-----------------------------" << endl << endl;

	// Draw contours
	// Documentation for drawing contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours#drawcontours
	drawContours(contour_output, contours, maxind, Scalar(0, 0, 255), 2, 8, hierarchy);

	// Show in a window
	namedWindow("Contours2", CV_WINDOW_AUTOSIZE);
	imshow("Contours2", contour_output);
}

