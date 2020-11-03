/*
CS585 Image and Video Computing Spring 2020
Project: License Plate Detection
Team Members: Ai Hue Nguyen, Juan Santana
--------------
Method 3: Machine Learning / KNN Algorithm
--------------
*/

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<sstream>

// global variables
//units in pixels
const int contour_area = 200; 

const int width = 20;
const int height = 30;


class DataContour {
public:

	// contour
	std::vector<cv::Point> contour; 
	// bounding rectangle for contours
	cv::Rect boundingRect;
	// area of contour
	float area;                             

	//Function to help reduce contours down to contours of characters
	bool contourValidity() {
		if (area < contour_area) return false;
		return true;
	}

	// sort contours from left to right
	static bool sortLeftToRight(const DataContour& cwdLeft, const DataContour& cwdRight) {
		return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);
	}

};


int main() {

	// Image can be switched out for other images from the dataset
	cv::Mat image = cv::imread("Fifty_States_License_Plates/Guam.jpg");

	cv::Mat grayscale_img;
	cv::Mat blurred_img;
	cv::Mat threshold_img;
	cv::Mat threshold_img_copy;

	std::vector<DataContour> datacontours;
	std::vector<DataContour> validdatacontours;

	// read in training classifications
	cv::Mat classification_ints;

	// open the classifications file
	cv::FileStorage fsClassifications("classification.xml", cv::FileStorage::READ);

	if (fsClassifications.isOpened() == false) {
		std::cout << "error, unable to open training classifications file, exiting program\n\n";
		return(0);
	}

	// read classifications section into variable
	fsClassifications["classifications"] >> classification_ints; 
	// close file
	fsClassifications.release();                                


	// read in training images
	cv::Mat matTrainingImagesAsFlattenedFloats;

	// open the training image file
	cv::FileStorage fsTrainingImages("image.xml", cv::FileStorage::READ);

	if (fsTrainingImages.isOpened() == false) {
		std::cout << "ERROR, can't open training image file, exiting program\n\n";
		return(0);
	}

	// read images section into Mat training images variable and close traning image file
	fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;
	fsTrainingImages.release();

	// start up OpenCV's KNN object
	cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());

	// Start training KNN object
	kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, classification_ints);


	// Error message if image can't open
	if (image.empty()) {
		std::cout << "ERROR: Image can't be opened\n\n";
		return(0);
	}


	// convert to grayscale
	cv::cvtColor(image, grayscale_img, CV_BGR2GRAY);


	// blur the grayscale image
	cv::GaussianBlur(grayscale_img, blurred_img, cv::Size(5, 5), 0);

	// filter blur image from grayscale to black and white
	cv::adaptiveThreshold(blurred_img, threshold_img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

	// make a copy of the thresh image since findContours modifies the image
	threshold_img_copy = threshold_img.clone();

	// declare contours and hierarchy vectors
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(threshold_img_copy, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// for each contour, assign it to contour with data, get the bounding rec, calculate the area, and add the object to list
	for (int i = 0; i < contours.size(); i++) {
		DataContour DataContour;
		DataContour.contour = contours[i];
		DataContour.boundingRect = cv::boundingRect(DataContour.contour);
		DataContour.area = cv::contourArea(DataContour.contour);
		datacontours.push_back(DataContour);
	}

	// for all countours, check if valid to add to valid contour list 
	for (int i = 0; i < datacontours.size(); i++) {
		if (datacontours[i].contourValidity()) {
			validdatacontours.push_back(datacontours[i]);
		}
	}

	// sort contours from left to right
	std::sort(validdatacontours.begin(), validdatacontours.end(), DataContour::sortLeftToRight);

	// this variable will have the final character sequence at the end 
	std::string answer_string;        

	// for each contour, draw the contour rectangle around the char
	for (int i = 0; i < validdatacontours.size(); i++) {
		cv::rectangle(image, validdatacontours[i].boundingRect, cv::Scalar(255, 0, 0), 2);

		// get roi image of bounding rect
		cv::Mat roi = threshold_img(validdatacontours[i].boundingRect);

		cv::Mat roi_resized;
		cv::resize(roi, roi_resized, cv::Size(width, height));

		cv::Mat roiFloat;

		// convert Mat to float in order to call find_nearest
		roi_resized.convertTo(roiFloat, CV_32FC1);

		cv::Mat roi_flat_float = roiFloat.reshape(1, 1);

		cv::Mat currchar(0, 0, CV_32F);

		// KNN Function Call
		kNearest->findNearest(roi_flat_float, 1, currchar);

		float flt_currchar = (float)currchar.at<float>(0, 0);

		// append current char to string
		answer_string = answer_string + char(int(flt_currchar));
	}

	// show string answer
	std::cout << "\n\n" << "Characters recognized from License Plate Image: " << answer_string << "\n\n";

	// show input image with boxes drawn around found digits
	cv::imshow("Output Image", image);

	cv::waitKey(0);

	return(0);
}