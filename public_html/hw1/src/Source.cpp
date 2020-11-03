#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main()
{
	// Code to display original image 
	Mat image;
	image = imread("selfie.jpg", IMREAD_COLOR); // Read the file

	if (image.empty())                      // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return 0;
	}

	//IMPORTANT NOTE: The color order here in OpenCV is BGR, not RGB !!!

	namedWindow("Selfie Original Version", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Selfie Original Version", image);                // Show our image inside it.
	imwrite("Selfie_Original_Version.bmp", image);             // Creates jpg file of image displayed
	

	// Code to convert orginal image to Grayscale
	int Rows = image.rows;
	int Cols = image.cols;
	Mat grayImage = Mat(Rows, Cols, CV_8UC1);         

	for (int row = 0; row < Rows; row++) {
		for (int col = 0; col < Cols; col++)
		{
			Vec3b intensity = image.at<Vec3b>(row, col);

			// Grayscale Conversion Method: Average R, G, B: (R+G+B)/3; in this case it's (B+G+R)/3
			int avg = (0.1 * intensity.val[0] + 0.6 * intensity.val[1] + 0.3 * intensity.val[2]);

			grayImage.at<uchar>(row, col) = avg;
		}
	}

	namedWindow("Selfie Gray Version", WINDOW_AUTOSIZE);       // Create a window for display.
	imshow("Selfie Gray Version", grayImage);                  // Show our image inside it.
	// OpenCV has problems trying to save the image in jpg format but can save it in a bmp (bitmap image) file
	imwrite("Selfie_Gray_Version.bmp", grayImage);             // Creates bmp file of image displayed
    
	// Code to flip the gray image horizontally, i.e. left to right, right to left.
	Mat horizontally_flip_image = Mat(Rows, Cols, CV_8UC1);
	Mat copyofgrayImage = grayImage;

	for (int row = 0; row < Rows; row++) {
		// Loop goes to half of the columns of the image and for each pixel on one side, 
		for (int col = 0; col < Cols/2; col++)
		{
			
			// Now find the corresponding pixel on the other half of the image and exchange the intensity values
			uchar temp = copyofgrayImage.at<uchar>(row, col);
			horizontally_flip_image.at<uchar>(row, col) = copyofgrayImage.at<uchar>(row, Cols - 1 - col);
			horizontally_flip_image.at<uchar>(row, Cols - 1 - col) = temp;
		}
	}

	namedWindow("Selfie Horizontal Flip Version", WINDOW_AUTOSIZE);      // Create a window for display.
	imshow("Selfie Horizontal Flip Version", horizontally_flip_image);   // Show our image inside it.
	// OpenCV has problems trying to save the image in jpg format but can save it in a bmp (bitmap image) file
	imwrite("Selfie_Horizontal_Flip_Version.bmp", horizontally_flip_image);             // Creates bmp file of image displayed

	// Code to create the negative version of the gray image
	Mat negative_of_gray_image = Mat(Rows, Cols, CV_8UC1);

	for (int row = 0; row < Rows; row++) {
		for (int col = 0; col < Cols; col++)
		{
			// Since 255 is the highest intensity value, we go to each pixel 
			//and finds it's complementary instensity ( 255 - original instensity ) thus find its negative version of the orginal pixel   
			negative_of_gray_image.at<uchar>(row, col) = 255 - grayImage.at<uchar>(row, col);
		}
	}

	namedWindow("Selfie Negative Gray Version", WINDOW_AUTOSIZE);      // Create a window for display.
	imshow("Selfie Negative Gray Version", negative_of_gray_image);   // Show our image inside it.
	// OpenCV has problems trying to save the image in jpg format but can save it in a bmp (bitmap image) file
	imwrite("Selfie_Negative_Gray_Version.bmp", negative_of_gray_image);  // Creates bmp file of image displayed

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}