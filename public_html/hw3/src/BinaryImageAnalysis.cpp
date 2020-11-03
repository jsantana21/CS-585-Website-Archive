/**
CS585 Image and Video Computing
HW 3: Object Shape Analysis and Segmentation Programming Assignment
--------------
Problem 1: Binary Image Analysis
--------------
Your data for this part are binary images of hands and a stained tissue sample of a breast cancer tumor.
--------------
Tasks:
1. Implement a connected component labeling algorithm and apply it to the data below. Show your results by displaying the detected components/objects with different colors. The recursive connected component labeling algorithm sometimes does not work properly if the image regions are large. See below for an alternate variant of this algorithm. If you have trouble implementing connected component labeling, you may use an OpenCV library function, but you will be assessed a small 5 point penalty.
2. If your output contains a large number of components, apply a technique to reduce the number of components, e.g., filtering by component size, or erosion.
3. Implement the boundary following algorithm and apply it for the relevant regions/objects.
4. For each relevant region/object, compute the area, orientation, and circularity (Emin/Emax). Also, identify and count the boundary pixels of each region, and compute compactness, the ratio of the area to the perimeter.
5. Implement a skeleton finding algorithm. Apply it to the relevant regions/objects.
--------------
*/

#include <iostream>
#include <string>
#include <math.h> 

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

//FUNCTION HEADERS
//void floodFill(Mat& src, Mat& dst);
void boundary(Mat& src, Mat& dst);
void mySkinDetect(Mat& src, Mat& dst);
double* calculations(Mat& src);
void skeleton(Mat& src, Mat& dst);

int rmin = 45, rmax = 192, bmin = 18, bmax = 147, gmin = 76, gmax = 151;


int main()
{
	// Read all 4 files
	Mat image1;
	image1 = imread("open_fist-bw.png", IMREAD_COLOR); 
	Mat image2;
	image2 = imread("open-bw-full.png", IMREAD_COLOR); 
	Mat image3;
	image3 = imread("open-bw-partial.png", IMREAD_COLOR); 
	Mat image4;
	image4 = imread("tumor-fold.png", IMREAD_COLOR);

	// Create a window for display for each image and show the images
	// Image 1
	//namedWindow("Open-Fist-BW Image", WINDOW_AUTOSIZE); 
	//imshow("Open-Fist-BW Image", image1);        
	//Image 2
	//namedWindow("Open-BW-Full Image", WINDOW_AUTOSIZE); 
	//imshow("Open-BW-Full Image", image2);      
	//Image 3
	//namedWindow("Open-BW-Partial Image", WINDOW_AUTOSIZE); 
	//imshow("Open-BW-Partial Image", image3);               
	//Image 4
	//namedWindow("Tumor-Fold Image", WINDOW_AUTOSIZE);
	//imshow("Tumor-Fold Image", image4);

	// Use threshold to create binary images of orginal images & use floodFill on binary image to color in the hands (and tumor)
	Mat image1_th; Mat image2_th; Mat image3_th; Mat image4_th;
	threshold(image1, image1_th, 220, 255, THRESH_BINARY);
	threshold(image2, image2_th, 220, 255, THRESH_BINARY);
	threshold(image3, image3_th, 220, 255, THRESH_BINARY);
	threshold(image4, image4_th, 220, 255, THRESH_BINARY);
	//imshow("image1th", image1_th);
	//imshow("image2th", image2_th);
	//imshow("image3th", image3_th);
	//imshow("image4th", image4_th);
	
	// initialize structuring element for morphological operations
	int erosion_size = 4;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
	//perform erosions and dilations to reduce the blur on the image boundary and remove noise
	// Opening on image1
	dilate(image1_th, image1_th, element);
	erode(image1_th, image1_th, element);
	//dilate(image1_th, image1_th, element);
	// Opening on image2
	erode(image2_th, image2_th, element);
	dilate(image2_th, image2_th, element);
	// Opening on image3
	int size3 = 0.25;
	Mat element3 = getStructuringElement(MORPH_RECT, Size(2 * size3 + 1, 2 * size3 + 1), Point(size3, size3));
	erode(image3_th, image3_th, element3);
	dilate(image3_th, image3_th, element3);
	dilate(image3_th, image3_th, element3);
	// Opening on image4
	int size4 = 0.5;
	Mat element4 = getStructuringElement(MORPH_RECT, Size(2 * size4 + 1, 2 * size4 + 1), Point(size4, size4));
	erode(image4_th, image4_th, element4);
	erode(image4_th, image4_th, element4);
	erode(image4_th, image4_th, element4);
	dilate(image4_th, image4_th, element4);

	//imshow("image1th", image1_th);
	//imshow("image2th", image2_th);
	//imshow("image3th", image3_th);
	//imshow("image4th", image4_th);

	Mat image1_floodfill = image1_th.clone();
	Mat image2_floodfill = image2_th.clone();
	Mat image3_floodfill = image3_th.clone();
	Mat image4_floodfill = image4_th.clone();

	// Labelled Component Images 
	floodFill(image1_floodfill, cv::Point(80, 160), Scalar(255,0, 0));//Point(80,160) is a pixel in the fist in the image
	floodFill(image1_floodfill, cv::Point(300, 180), Scalar(255, 0, 0));// Point(300, 180) is a pixel in the open hand in the image
	floodFill(image1_floodfill, cv::Point(1, 1), Scalar(255, 255, 255)); // white out unnecessary parts
	floodFill(image1_floodfill, cv::Point(3, 65), Scalar(255, 255, 255));
	floodFill(image1_floodfill, cv::Point(3, 80), Scalar(255, 255, 255));
	imshow("Open-Fist-BW Image Labelled", image1_floodfill);
	floodFill(image2_floodfill, cv::Point(300, 100), Scalar(255, 0, 0)); //Point (300,100) is a pixel in the open hand in the image
	imshow("Open-BW-Full Image Labelled", image2_floodfill);
	floodFill(image3_floodfill, cv::Point(330, 70), Scalar(255, 0, 0)); //Point (330,70) is a pixel in the black area of the open hand in the image
	floodFill(image3_floodfill, cv::Point(200, 125), Scalar(255, 0, 0));
	floodFill(image3_floodfill, cv::Point(380, 280), Scalar(255, 255, 255)); //white out unnecessary parts
	floodFill(image3_floodfill, cv::Point(10, 10), Scalar(255, 255, 255));
	floodFill(image3_floodfill, cv::Point(10, 260), Scalar(255, 255, 255));
	imshow("Open-BW-Partial Image Labelled", image3_floodfill);
	floodFill(image4_floodfill, cv::Point(585, 325), Scalar(255, 0, 0)); //Point (585,325) is a pixel in the black area of the open hand in the image
	floodFill(image4_floodfill, cv::Point(500, 600), Scalar(255, 255, 255)); //white out unnecessary parts
	floodFill(image4_floodfill, cv::Point(170, 200), Scalar(255, 255, 255));
	floodFill(image4_floodfill, cv::Point(650, 590), Scalar(255, 255, 255));
	floodFill(image4_floodfill, cv::Point(970, 360), Scalar(255, 255, 255));
	floodFill(image4_floodfill, cv::Point(45, 570), Scalar(255, 255, 255));
	floodFill(image4_floodfill, cv::Point(450, 320), Scalar(255, 255, 255));
	floodFill(image4_floodfill, cv::Point(510, 330), Scalar(255, 255, 255));
	floodFill(image4_floodfill, cv::Point(830, 390), Scalar(255, 255, 255));
	floodFill(image4_floodfill, cv::Point(960, 590), Scalar(255, 255, 255));
	floodFill(image4_floodfill, cv::Point(510, 230), Scalar(255, 255, 255));
	imshow("Tumor-Fold Image Labelled", image4_floodfill);

	//imwrite("./fist.jpg", image1_floodfill);

	//Boundary Trace Images
	Mat image1_boundary = image1_floodfill.clone();
	Mat image2_boundary = image2_floodfill.clone();
	Mat image3_boundary = image3_floodfill.clone();
	Mat image4_boundary = image4_floodfill.clone();

	//boundary(image1_floodfill, image1_boundary);
	//imshow("Open-Fist-BW Image Boundary", image1_boundary);

	//boundary(image2_floodfill, image2_boundary);
	//imshow("Open-BW-Full Image Boundary", image2_boundary);

	//boundary(image3_floodfill, image3_boundary);
	//imshow("Open-BW-Partial Image Boundary", image3_boundary);

	//boundary(image4_floodfill, image4_boundary);
	//imshow("Tumor-Fold Image Boundary", image4_boundary);

	//Calculations for all images
	//Image 1
	cout << "Image 1: Open-Fist-BW Image" << endl;
	cout << "Objects: 2" << endl;
	cout << "Area: " << calculations(image1_floodfill)[0] << endl;
	cout << "Orientation: " << calculations(image1_floodfill)[1] << " degrees" << endl;
	cout << "Circularity: " << calculations(image1_floodfill)[2] << endl;
	cout << "Compactness: " << calculations(image1_floodfill)[3] << endl;
	//cout << "Object: 2" << endl;
	//cout << "Area: " << endl;
	//cout << "Orientation: " << endl;
	//cout << "Circularity: " << endl;
	//cout << "Compactness: " << endl;
	
	//Image 2
	cout << "\nImage 2: Open-BW-Full Image" << endl;
	cout << "Object: 1" << endl;
	cout << "Area: " << calculations(image2_floodfill)[0] << endl;
	cout << "Orientation: " << calculations(image2_floodfill)[1] << " degrees"<< endl;
	cout << "Circularity: " << calculations(image2_floodfill)[2] << endl;
	cout << "Compactness: " << calculations(image2_floodfill)[3] << endl;
	//Image 3
	cout << "\nImage 3: Open-BW-Partial Image" << endl;
	cout << "Object: 1" << endl;
	cout << "Area: " << calculations(image3_floodfill)[0] << endl;
	cout << "Orientation: " << calculations(image3_floodfill)[1] << " degrees" << endl;
	cout << "Circularity: " << calculations(image3_floodfill)[2] << endl;
	cout << "Compactness: " << calculations(image3_floodfill)[3] << endl;
	//Image 4
	cout << "\nImage 4: Tumor-Fold Image" << endl;
	cout << "Object: 1" << endl;
	cout << "Area: " << calculations(image4_floodfill)[0] << endl;
	cout << "Orientation: " << calculations(image4_floodfill)[1] << " degrees" << endl;
	cout << "Circularity: " << calculations(image4_floodfill)[2] << endl;
	cout << "Compactness: " << calculations(image4_floodfill)[3] << endl;

	//Skeleton Images
	Mat image1_skeleton = image1.clone();
	Mat image2_skeleton = image2.clone();
	Mat image3_skeleton = image3.clone();
	Mat image4_skeleton = image4.clone();

	skeleton(image1_floodfill, image1_skeleton);
	imshow("Open-Fist-BW Image Skeleton", image1_skeleton);

	skeleton(image2_floodfill, image2_skeleton);
	imshow("Open-BW-Full Image Skeleton", image2_skeleton);

	skeleton(image3_floodfill, image3_skeleton);
	imshow("Open-BW-Partial Image Skeleton", image3_skeleton);

	skeleton(image4_floodfill, image4_skeleton);
	imshow("Tumor-Fold Image", image4_skeleton);


	waitKey(0); // Wait for a keystroke in the window
	return 0;
}


void boundary(Mat& src, Mat& dst) {
	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat src_gray;

	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));
	
	/// Detect edges using canny
	Canny( src_gray, canny_output, 100, 100*2, 3 );
	/// Find contours
	findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	/// Draw contours
	Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
     {
		Scalar color = Scalar(255,0,0);
		drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
     }
	drawing.copyTo(dst);
	/// Show in a window
	//namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	//imshow( "Contours", drawing );
}

double* calculations(Mat& src) {

	static double  ans[4];

	Mat thr, gray;

	// convert image to grayscale
	cvtColor(src, gray, COLOR_BGR2GRAY);

	// convert grayscale to binary image
	threshold(gray, thr, 100, 255, THRESH_BINARY);

	// find moments of the image
	Moments m = moments(thr, true);

	// Zero Image Moment = Area of Object in Image
	double area = m.m00;
	ans[0] = area;

	// Orientation formula = 0.5 arctan( (2*mu11) / (mu20 - mu02))
	double orientation = 0.5 * atan((2*m.mu11)/(m.mu20 - m.mu02));
	//radians to degress
	orientation = orientation * (180/3.14);
	ans[1] = orientation;

	double a = m.mu20 + m.mu02;
	double b = m.mu11;
	double c = m.mu20 - m.mu02;

	//Circularity = Emin / Emax
	double emin = (a / 2) - ((c / 2) * (c / sqrt(pow(c, 2) - pow(b, 2)))) - ((b / 2) *(b / sqrt(pow(c, 2) - pow(b, 2))));
	double emax = (a / 2) + ((c / 2) * (c / sqrt(pow(c, 2) - pow(b, 2)))) + ((b / 2) *(b / sqrt(pow(c, 2) - pow(b, 2))));
	double circularity = emin / emax;
	ans[2] = circularity;

	//Find perimeter for compactness
	vector<vector<Point> > contour;
	vector<Vec4i> hierarchy;
	//findContours(src, contour, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	// ???
	double perimeter = 100000;//arcLength(contour[0], true);

	// Compactness = Perimeter / Area
	double compactness = perimeter / area;
	ans[3] = compactness;

	return ans;
}


void skeleton(Mat& src, Mat& dst) {

	//Process src image in grayscale and chnage it to a binary image w/ threshold
	threshold(src, src, 127, 255, THRESH_BINARY_INV);
	//temporary image in order to store intermediate computations in loop
	Mat temp;
	Mat eroded;
	Mat img = src.clone();
	Mat skeleton = dst.clone();

	//Declare structuring element for morphological operations (3x3 cross-shaped structure element)
	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));

	//Check if there is at least one pixel remaining;Operations done in-place when possible
	bool done;
	do
	{
		erode(img, eroded, element);
		dilate(eroded, temp, element);
		subtract(img, temp, temp);
		bitwise_or(skeleton, temp, skeleton);
		eroded.copyTo(img);

		double max;
		cv::minMaxLoc(img, 0, &max);
		done = (max == 0);
	} while (!done);

	skeleton.copyTo(dst);

	//imshow("Skeleton Image", skeleton);

}

//Function that detects whether a pixel belongs to the skin based on RGB values
void mySkinDetect(Mat& src, Mat& dst) {
	//Surveys of skin color modeling and detection techniques:
	//Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
	//Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.

	inRange(src, Scalar(bmin, gmin, rmin), Scalar(bmax, gmax, rmax), dst);

}