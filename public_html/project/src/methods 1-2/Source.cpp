/*
CS585 Image and Video Computing Spring 2020
Project: License Plate Detection
Team Members: Ai Hue Nguyen, Juan Santana
--------------
Methods 1 and 2
--------------
*/

#include <stdio.h>
#include <stdlib.h>
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

// Function declarations
void segmentNumbersAndLetters(Mat& img, Mat& binary_img);
vector<Mat> labelObjects(Mat& img, Mat& labelled_img);
vector<Point> findLargestContour(Mat& img, Mat& labelled_img, Mat& templ);
vector<Point> MatchingMethod(Mat &img_bw, Mat &img_color, Mat &templ, string name);
int myMaxVal(double* array, int size);
int myMinVal(double* array, int size);
double correlation(Mat &image1, Mat &image2);

// Initialize templates
Mat templ0, templ1, templ2, templ3, templ4, templ5, templ6, templ7, templ8, templ9;
Mat templ10, templ11, templ12, templ13, templ14, templ15, templ16, templ17, templ18, templ19;
Mat templ20, templ21, templ22, templ23, templ24, templ25, templ26, templ27, templ28, templ29;
Mat templ30, templ31, templ32, templ33, templ34, templ35;

int main()
{
	// Binarize source license plate image in order to segment numbers and letters from it
	Mat input_img = imread("Fifty_States_License_Plates/DistrictofColumbia.jpg", 1);
	//Mat input_img = imread("Fifty_States_License_Plates/Arizona.jpg", 1);
	//Mat input_img = imread("Fifty_States_License_Plates/Delaware.jpg", 1);
	//Mat input_img = imread("Fifty_States_License_Plates/Oregon.jpg", 1);
	Mat binarized_img, labelled_img;
	segmentNumbersAndLetters(input_img, binarized_img);

	// Display results of segmentation
	namedWindow("Source Image", 1);
	imshow("Source Image", input_img);
	imshow("Binarized Source Image", binarized_img);
	
	// Initialize templates
	Mat bw_template0, bw_template1, bw_template2, bw_template3, bw_template4;
	Mat bw_template5, bw_template6, bw_template7, bw_template8, bw_template9;
	Mat bw_template10, bw_template11, bw_template12, bw_template13, bw_template14;
	Mat bw_template15, bw_template16, bw_template17, bw_template18, bw_template19;
	Mat bw_template20, bw_template21, bw_template22, bw_template23, bw_template24;
	Mat bw_template25, bw_template26, bw_template27, bw_template28, bw_template29;
	Mat bw_template30, bw_template31, bw_template32, bw_template33, bw_template34, bw_template35;

	Mat labeled_template0, labeled_template1, labeled_template2, labeled_template3, labeled_template4;
	Mat labeled_template5, labeled_template6, labeled_template7, labeled_template8, labeled_template9;
	Mat labeled_template10, labeled_template11, labeled_template12, labeled_template13, labeled_template14;
	Mat labeled_template15, labeled_template16, labeled_template17, labeled_template18, labeled_template19;
	Mat labeled_template20, labeled_template21, labeled_template22, labeled_template23, labeled_template24;
	Mat labeled_template25, labeled_template26, labeled_template27, labeled_template28, labeled_template29;
	Mat labeled_template30, labeled_template31, labeled_template32, labeled_template33, labeled_template34, labeled_template35;

	// Read all of the template images (digits 0-9, and letters A-Z) 
	Mat template0 = imread("0.jpg", CV_32FC1);
	Mat template1 = imread("1.jpg", CV_32FC1);
	Mat template2 = imread("2.jpg", CV_32FC1);
	Mat template3 = imread("3.jpg", CV_32FC1);
	Mat template4 = imread("4.jpg", CV_32FC1);
	Mat template5 = imread("5.jpg", CV_32FC1);
	Mat template6 = imread("6.jpg", CV_32FC1);
	Mat template7 = imread("7.jpg", CV_32FC1);
	Mat template8 = imread("8.jpg", CV_32FC1);
	Mat template9 = imread("9.jpg", CV_32FC1);
	Mat template10 = imread("A.jpg", CV_32FC1);
	Mat template11 = imread("B.jpg", CV_32FC1);
	Mat template12 = imread("C.jpg", CV_32FC1);
	Mat template13 = imread("D.jpg", CV_32FC1);
	Mat template14 = imread("E.jpg", CV_32FC1);
	Mat template15 = imread("F.jpg", CV_32FC1);
	Mat template16 = imread("G.jpg", CV_32FC1);
	Mat template17 = imread("H.jpg", CV_32FC1);
	Mat template18 = imread("I.jpg", CV_32FC1);
	Mat template19 = imread("J.jpg", CV_32FC1);
	Mat template20 = imread("K.jpg", CV_32FC1);
	Mat template21 = imread("L.jpg", CV_32FC1);
	Mat template22 = imread("M.jpg", CV_32FC1);
	Mat template23 = imread("N.jpg", CV_32FC1);
	Mat template24 = imread("O.jpg", CV_32FC1);
	Mat template25 = imread("P.jpg", CV_32FC1);
	Mat template26 = imread("Q.jpg", CV_32FC1);
	Mat template27 = imread("R.jpg", CV_32FC1);
	Mat template28 = imread("S.jpg", CV_32FC1);
	Mat template29 = imread("T.jpg", CV_32FC1);
	Mat template30 = imread("U.jpg", CV_32FC1);
	Mat template31 = imread("V.jpg", CV_32FC1);
	Mat template32 = imread("W.jpg", CV_32FC1);
	Mat template33 = imread("X.jpg", CV_32FC1);
	Mat template34 = imread("Y.jpg", CV_32FC1);
	Mat template35 = imread("Z.jpg", CV_32FC1);

	// Segment the character from each template image 
	segmentNumbersAndLetters(template0, bw_template0);
	segmentNumbersAndLetters(template1, bw_template1);
	segmentNumbersAndLetters(template2, bw_template2);
	segmentNumbersAndLetters(template3, bw_template3);
	segmentNumbersAndLetters(template4, bw_template4);
	segmentNumbersAndLetters(template5, bw_template5);
	segmentNumbersAndLetters(template6, bw_template6);
	segmentNumbersAndLetters(template7, bw_template7);
	segmentNumbersAndLetters(template8, bw_template8);
	segmentNumbersAndLetters(template9, bw_template9);
	segmentNumbersAndLetters(template10, bw_template10);
	segmentNumbersAndLetters(template11, bw_template11);
	segmentNumbersAndLetters(template12, bw_template12);
	segmentNumbersAndLetters(template13, bw_template13);
	segmentNumbersAndLetters(template14, bw_template14);
	segmentNumbersAndLetters(template15, bw_template15);
	segmentNumbersAndLetters(template16, bw_template16);
	segmentNumbersAndLetters(template17, bw_template17);
	segmentNumbersAndLetters(template18, bw_template18);
	segmentNumbersAndLetters(template19, bw_template19);
	segmentNumbersAndLetters(template20, bw_template20);
	segmentNumbersAndLetters(template21, bw_template21);
	segmentNumbersAndLetters(template22, bw_template22);
	segmentNumbersAndLetters(template23, bw_template23);
	segmentNumbersAndLetters(template24, bw_template24);
	segmentNumbersAndLetters(template25, bw_template25);
	segmentNumbersAndLetters(template26, bw_template26);
	segmentNumbersAndLetters(template27, bw_template27);
	segmentNumbersAndLetters(template28, bw_template28);
	segmentNumbersAndLetters(template29, bw_template29);
	segmentNumbersAndLetters(template30, bw_template30);
	segmentNumbersAndLetters(template31, bw_template31);
	segmentNumbersAndLetters(template32, bw_template32);
	segmentNumbersAndLetters(template33, bw_template33);
	segmentNumbersAndLetters(template34, bw_template34);
	segmentNumbersAndLetters(template35, bw_template35);

	// Find six largest contours in source image
	vector<Mat> characters;
	characters = labelObjects(binarized_img, labelled_img);

	// vector to hold outputs from first method
	vector<char> outputs;

	// vector to hold outputs from second method
	vector<char> outputs2;

	// for each of the six largest contours, finds the best template match 
	for (int i = 0; i < characters.size(); i++) {

		Mat src;
		src = characters[i];
		Mat gray;
		gray = Mat(src.rows, src.cols, CV_8UC3);
		cvtColor(src, gray, CV_BGR2GRAY);
		Mat gray_src;
		threshold(gray, gray_src, 50, 255, CV_THRESH_BINARY);

		// First method (based on code from HW 2) 

		vector<Point> max_cont0, max_cont1, max_cont2, max_cont3, max_cont4;
		vector<Point> max_cont5, max_cont6, max_cont7, max_cont8, max_cont9;
		vector<Point> max_cont10, max_cont11, max_cont12, max_cont13, max_cont14;
		vector<Point> max_cont15, max_cont16, max_cont17, max_cont18, max_cont19;
		vector<Point> max_cont20, max_cont21, max_cont22, max_cont23, max_cont24;
		vector<Point> max_cont25, max_cont26, max_cont27, max_cont28, max_cont29;
		vector<Point> max_cont30, max_cont31, max_cont32, max_cont33, max_cont34, max_cont35;

		max_cont0 = findLargestContour(bw_template0, labeled_template0, templ0);
		max_cont1 = findLargestContour(bw_template1, labeled_template1, templ1);
		max_cont2 = findLargestContour(bw_template2, labeled_template2, templ2);
		max_cont3 = findLargestContour(bw_template3, labeled_template3, templ3);
		max_cont4 = findLargestContour(bw_template4, labeled_template4, templ4);
		max_cont5 = findLargestContour(bw_template5, labeled_template5, templ5);
		max_cont6 = findLargestContour(bw_template6, labeled_template6, templ6);
		max_cont7 = findLargestContour(bw_template7, labeled_template7, templ7);
		max_cont8 = findLargestContour(bw_template8, labeled_template8, templ8);
		max_cont9 = findLargestContour(bw_template9, labeled_template9, templ9);
		max_cont10 = findLargestContour(bw_template10, labeled_template10, templ10);
		max_cont11 = findLargestContour(bw_template11, labeled_template11, templ11);
		max_cont12 = findLargestContour(bw_template12, labeled_template12, templ12);
		max_cont13 = findLargestContour(bw_template13, labeled_template13, templ13);
		max_cont14 = findLargestContour(bw_template14, labeled_template14, templ14);
		max_cont15 = findLargestContour(bw_template15, labeled_template15, templ15);
		max_cont16 = findLargestContour(bw_template16, labeled_template16, templ16);
		max_cont17 = findLargestContour(bw_template17, labeled_template17, templ17);
		max_cont18 = findLargestContour(bw_template18, labeled_template18, templ18);
		max_cont19 = findLargestContour(bw_template19, labeled_template19, templ19);
		max_cont20 = findLargestContour(bw_template20, labeled_template20, templ20);
		max_cont21 = findLargestContour(bw_template21, labeled_template21, templ21);
		max_cont22 = findLargestContour(bw_template22, labeled_template22, templ22);
		max_cont23 = findLargestContour(bw_template23, labeled_template23, templ23);
		max_cont24 = findLargestContour(bw_template24, labeled_template24, templ24);
		max_cont25 = findLargestContour(bw_template25, labeled_template25, templ25);
		max_cont26 = findLargestContour(bw_template26, labeled_template26, templ26);
		max_cont27 = findLargestContour(bw_template27, labeled_template27, templ27);
		max_cont28 = findLargestContour(bw_template28, labeled_template28, templ28);
		max_cont29 = findLargestContour(bw_template29, labeled_template29, templ29);
		max_cont30 = findLargestContour(bw_template30, labeled_template30, templ30);
		max_cont31 = findLargestContour(bw_template31, labeled_template31, templ31);
		max_cont32 = findLargestContour(bw_template32, labeled_template32, templ32);
		max_cont33 = findLargestContour(bw_template33, labeled_template33, templ33);
		max_cont34 = findLargestContour(bw_template34, labeled_template34, templ34);
		max_cont35 = findLargestContour(bw_template35, labeled_template35, templ35);

		// Based on code from HW 2
		// ----------
		vector<Point> cont0, cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8, cont9;
		vector<Point> cont10, cont11, cont12, cont13, cont14, cont15, cont16, cont17, cont18, cont19;
		vector<Point> cont20, cont21, cont22, cont23, cont24, cont25, cont26, cont27, cont28, cont29;
		vector<Point> cont30, cont31, cont32, cont33, cont34, cont35;

		//Perform matching method through the shapes
		cont0 = MatchingMethod(gray_src, src, templ1, "Zero");
		cont1 = MatchingMethod(gray_src, src, templ1, "One");
		cont2 = MatchingMethod(gray_src, src, templ2, "Two");
		cont3 = MatchingMethod(gray_src, src, templ3, "Three");
		cont4 = MatchingMethod(gray_src, src, templ4, "Four");
		cont5 = MatchingMethod(gray_src, src, templ5, "Five");
		cont6 = MatchingMethod(gray_src, src, templ6, "Six");
		cont7 = MatchingMethod(gray_src, src, templ7, "Seven");
		cont8 = MatchingMethod(gray_src, src, templ8, "Eight");
		cont9 = MatchingMethod(gray_src, src, templ9, "Nine");
		cont10 = MatchingMethod(gray_src, src, templ10, "A");
		cont11 = MatchingMethod(gray_src, src, templ11, "B");
		cont12 = MatchingMethod(gray_src, src, templ12, "C");
		cont13 = MatchingMethod(gray_src, src, templ13, "D");
		cont14 = MatchingMethod(gray_src, src, templ14, "E");
		cont15 = MatchingMethod(gray_src, src, templ15, "F");
		cont16 = MatchingMethod(gray_src, src, templ16, "G");
		cont17 = MatchingMethod(gray_src, src, templ17, "H");
		cont18 = MatchingMethod(gray_src, src, templ18, "I");
		cont19 = MatchingMethod(gray_src, src, templ19, "J");
		cont20 = MatchingMethod(gray_src, src, templ20, "K");
		cont21 = MatchingMethod(gray_src, src, templ21, "L");
		cont22 = MatchingMethod(gray_src, src, templ22, "M");
		cont23 = MatchingMethod(gray_src, src, templ23, "N");
		cont24 = MatchingMethod(gray_src, src, templ24, "O");
		cont25 = MatchingMethod(gray_src, src, templ25, "P");
		cont26 = MatchingMethod(gray_src, src, templ26, "Q");
		cont27 = MatchingMethod(gray_src, src, templ27, "R");
		cont28 = MatchingMethod(gray_src, src, templ28, "S");
		cont29 = MatchingMethod(gray_src, src, templ29, "T");
		cont30 = MatchingMethod(gray_src, src, templ30, "U");
		cont31 = MatchingMethod(gray_src, src, templ31, "V");
		cont32 = MatchingMethod(gray_src, src, templ32, "W");
		cont33 = MatchingMethod(gray_src, src, templ33, "X");
		cont34 = MatchingMethod(gray_src, src, templ34, "Y");
		cont35 = MatchingMethod(gray_src, src, templ35, "Z");

		double match[36];

		//Calculate how similar each correspondent template contour and character contour are

		match[0] = matchShapes(cont0, max_cont0, 1, 0);
		match[1] = matchShapes(cont1, max_cont1, 1, 0);
		match[2] = matchShapes(cont2, max_cont2, 1, 0);
		match[3] = matchShapes(cont3, max_cont3, 1, 0);
		match[4] = matchShapes(cont4, max_cont4, 1, 0);
		match[5] = matchShapes(cont5, max_cont5, 1, 0);
		match[6] = matchShapes(cont6, max_cont6, 1, 0);
		match[7] = matchShapes(cont7, max_cont7, 1, 0);
		match[8] = matchShapes(cont8, max_cont8, 1, 0);
		match[9] = matchShapes(cont9, max_cont9, 1, 0);
		match[10] = matchShapes(cont10, max_cont10, 1, 0);
		match[11] = matchShapes(cont11, max_cont11, 1, 0);
		match[12] = matchShapes(cont12, max_cont12, 1, 0);
		match[13] = matchShapes(cont13, max_cont13, 1, 0);
		match[14] = matchShapes(cont14, max_cont14, 1, 0);
		match[15] = matchShapes(cont15, max_cont15, 1, 0);
		match[16] = matchShapes(cont16, max_cont16, 1, 0);
		match[17] = matchShapes(cont17, max_cont17, 1, 0);
		match[18] = matchShapes(cont18, max_cont18, 1, 0);
		match[19] = matchShapes(cont19, max_cont19, 1, 0);
		match[20] = matchShapes(cont20, max_cont20, 1, 0);
		match[21] = matchShapes(cont21, max_cont21, 1, 0);
		match[22] = matchShapes(cont22, max_cont22, 1, 0);
		match[23] = matchShapes(cont23, max_cont23, 1, 0);
		match[24] = matchShapes(cont24, max_cont24, 1, 0);
		match[25] = matchShapes(cont25, max_cont25, 1, 0);
		match[26] = matchShapes(cont26, max_cont26, 1, 0);
		match[27] = matchShapes(cont27, max_cont27, 1, 0);
		match[28] = matchShapes(cont28, max_cont28, 1, 0);
		match[29] = matchShapes(cont29, max_cont29, 1, 0);
		match[30] = matchShapes(cont30, max_cont30, 1, 0);
		match[31] = matchShapes(cont31, max_cont31, 1, 0);
		match[32] = matchShapes(cont32, max_cont32, 1, 0);
		match[33] = matchShapes(cont33, max_cont33, 1, 0);
		match[34] = matchShapes(cont34, max_cont34, 1, 0);
		match[35] = matchShapes(cont35, max_cont35, 1, 0);

		//Find the minimum value of match array
		int min_value;
		min_value = myMinVal(match, 36);

		// Output of the program: Visualize the values of the matchShapes and output what shape is begin recognized by the algorithm
		// Uncomment code below to see some of the coefficients
		/*
		cout << "Coefficients: " << match[0] << " , " << match[1] << " , " << match[2] << " , " << match[3] << " , " << match[4] << " , " << match[5] << " , " << match[6] << " , " << match[7] << " , " << match[8] << " , " << match[9] << endl;
		if (max_value == 0) cout << "ZERO" << endl;
		if (max_value == 1) cout << "ONE" << endl;
		if (max_value == 2) cout << "TWO" << endl;
		if (max_value == 3) cout << "THREE" << endl;
		if (max_value == 4) cout << "FOUR" << endl;
		if (max_value == 5) cout << "FIVE" << endl;
		if (max_value == 6) cout << "SIX" << endl;
		if (max_value == 7) cout << "SEVEN" << endl;
		if (max_value == 8) cout << "EIGHT" << endl;
		if (max_value == 9) cout << "NINE" << endl;
		*/

		if (min_value == 0) outputs.push_back('0');
		if (min_value == 1) outputs.push_back('1');
		if (min_value == 2) outputs.push_back('2');
		if (min_value == 3) outputs.push_back('3');
		if (min_value == 4) outputs.push_back('4');
		if (min_value == 5) outputs.push_back('5');
		if (min_value == 6) outputs.push_back('6');
		if (min_value == 7) outputs.push_back('7');
		if (min_value == 8) outputs.push_back('8');
		if (min_value == 9) outputs.push_back('9');
		if (min_value == 10) outputs.push_back('A');
		if (min_value == 11) outputs.push_back('B');
		if (min_value == 12) outputs.push_back('C');
		if (min_value == 13) outputs.push_back('D');
		if (min_value == 14) outputs.push_back('E');
		if (min_value == 15) outputs.push_back('F');
		if (min_value == 16) outputs.push_back('G');
		if (min_value == 17) outputs.push_back('H');
		if (min_value == 18) outputs.push_back('I');
		if (min_value == 19) outputs.push_back('J');
		if (min_value == 20) outputs.push_back('K');
		if (min_value == 21) outputs.push_back('L');
		if (min_value == 22) outputs.push_back('M');
		if (min_value == 23) outputs.push_back('N');
		if (min_value == 24) outputs.push_back('0');
		if (min_value == 25) outputs.push_back('P');
		if (min_value == 26) outputs.push_back('Q');
		if (min_value == 27) outputs.push_back('R');
		if (min_value == 28) outputs.push_back('S');
		if (min_value == 29) outputs.push_back('T');
		if (min_value == 30) outputs.push_back('U');
		if (min_value == 31) outputs.push_back('V');
		if (min_value == 32) outputs.push_back('W');
		if (min_value == 33) outputs.push_back('X');
		if (min_value == 34) outputs.push_back('U');
		if (min_value == 35) outputs.push_back('Z');
 
		// Second method 
		double correlations[36];
		correlations[0] = correlation(gray_src, templ0);
		correlations[1] = correlation(gray_src, templ1);
		correlations[2] = correlation(gray_src, templ2);
		correlations[3] = correlation(gray_src, templ3);
		correlations[4] = correlation(gray_src, templ4);
		correlations[5] = correlation(gray_src, templ5);
		correlations[6] = correlation(gray_src, templ6);
		correlations[7] = correlation(gray_src, templ7);
		correlations[8] = correlation(gray_src, templ8);
		correlations[9] = correlation(gray_src, templ9);
		correlations[10] = correlation(gray_src, templ10);
		correlations[11] = correlation(gray_src, templ11);
		correlations[12] = correlation(gray_src, templ12);
		correlations[13] = correlation(gray_src, templ13);
		correlations[14] = correlation(gray_src, templ14);
		correlations[15] = correlation(gray_src, templ15);
		correlations[16] = correlation(gray_src, templ16);
		correlations[17] = correlation(gray_src, templ17);
		correlations[18] = correlation(gray_src, templ18);
		correlations[19] = correlation(gray_src, templ19);
		correlations[20] = correlation(gray_src, templ20);
		correlations[21] = correlation(gray_src, templ21);
		correlations[22] = correlation(gray_src, templ22);
		correlations[23] = correlation(gray_src, templ23);
		correlations[24] = correlation(gray_src, templ24);
		correlations[25] = correlation(gray_src, templ25);
		correlations[26] = correlation(gray_src, templ26);
		correlations[27] = correlation(gray_src, templ27);
		correlations[28] = correlation(gray_src, templ28);
		correlations[29] = correlation(gray_src, templ29);
		correlations[30] = correlation(gray_src, templ30);
		correlations[31] = correlation(gray_src, templ31);
		correlations[32] = correlation(gray_src, templ32);
		correlations[33] = correlation(gray_src, templ33);
		correlations[34] = correlation(gray_src, templ34);
		correlations[35] = correlation(gray_src, templ35);

		int max_value2;
		max_value2 = myMaxVal(correlations, 36);
		if (max_value2 == 0) outputs2.push_back('0');
		if (max_value2 == 1) outputs2.push_back('1');
		if (max_value2 == 2) outputs2.push_back('2');
		if (max_value2 == 3) outputs2.push_back('3');
		if (max_value2 == 4) outputs2.push_back('4');
		if (max_value2 == 5) outputs2.push_back('5');
		if (max_value2 == 6) outputs2.push_back('6');
		if (max_value2 == 7) outputs2.push_back('7');
		if (max_value2 == 8) outputs2.push_back('8');
		if (max_value2 == 9) outputs2.push_back('9');
		if (max_value2 == 10) outputs2.push_back('A');
		if (max_value2 == 11) outputs2.push_back('B');
		if (max_value2 == 12) outputs2.push_back('C');
		if (max_value2 == 13) outputs2.push_back('D');
		if (max_value2 == 14) outputs2.push_back('E');
		if (max_value2 == 15) outputs2.push_back('F');
		if (max_value2 == 16) outputs2.push_back('G');
		if (max_value2 == 17) outputs2.push_back('H');
		if (max_value2 == 18) outputs2.push_back('I');
		if (max_value2 == 19) outputs2.push_back('J');
		if (max_value2 == 20) outputs2.push_back('K');
		if (max_value2 == 21) outputs2.push_back('L');
		if (max_value2 == 22) outputs2.push_back('M');
		if (max_value2 == 23) outputs2.push_back('N');
		if (max_value2 == 24) outputs2.push_back('0');
		if (max_value2 == 25) outputs2.push_back('P');
		if (max_value2 == 26) outputs2.push_back('Q');
		if (max_value2 == 27) outputs2.push_back('R');
		if (max_value2 == 28) outputs2.push_back('S');
		if (max_value2 == 29) outputs2.push_back('T');
		if (max_value2 == 30) outputs2.push_back('U');
		if (max_value2 == 31) outputs2.push_back('V');
		if (max_value2 == 32) outputs2.push_back('W');
		if (max_value2 == 33) outputs2.push_back('X');
		if (max_value2 == 34) outputs2.push_back('U');
		if (max_value2 == 35) outputs2.push_back('Z');

		// Uncomment code below to see some of the coefficients 
		/*
		cout << "Zero: " << correlations[0] << " , One: " << correlations[1] << " ,  Two: " << correlations[2] << endl;
		cout << "Three: " << correlations[3] << " , Four: " << correlations[4] << " ,  Five: " << correlations[5] << endl;
		cout << "Six: " << correlations[6] << " , Seven: " << correlations[7] << " ,  Eight: " << correlations[8] << endl;
		cout << "Nine: " << correlations[9] << endl;
		*/

		imshow("Grayscale Source Image", gray_src);
	}

	// Results from first method 
	int size = outputs.size();
	char* str = new char[size];
	for (int j = 0; j < size; j++) {
		str[j] = outputs[j];
	}

	// Results from second method
	int size2 = outputs2.size();
	char* str2 = new char[size2];

	for (int j = 0; j < size2; j++) {
		str2[j] = outputs2[j];
	}

	cout << "License Plate Characters Detected (Method 1): " << str[0] << str[1] << str[2] << str[3] << str[4] << str[5] << endl;
	cout << "License Plate Characters Detected (Method 1): " << str2[0] << str2[1] << str2[2] << str2[3] << str2[4] << str2[5] << endl;

	// For testing
	//imshow("Templ1", templ1);
	//imshow("Template 5", templ5);
	//imshow("Template 10", templ10);
	//imshow("BW Template 35", bw_template35);

	waitKey(0);

	int x;
	cin >> x;
	return 0;
}

// Converts license plate image to binary in order to segment numbers and letters from it
void segmentNumbersAndLetters(Mat& img, Mat& binary_img) {
	// Convert input image to grayscale
	Mat dest = Mat::zeros(img.rows, img.cols, CV_8UC3);
	cvtColor(img, dest, CV_BGR2GRAY);

	// Convert grayscale image into binary image using thresholding (based on code from Lab5) 
	cv::threshold(dest, binary_img, 150, 255, 0);
}

// Finds the six largest contours in the binary license plate image, labels them, and returns them in the order of appearance left to right
vector<Mat> labelObjects(Mat& img, Mat& labelled_img) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	// Find contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
	findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Convert input image back into a color image so that circles in next part will show up as red 
	cvtColor(img, labelled_img, COLOR_GRAY2BGR);

	// Based on code in StackOverflow answer here: https://stackoverflow.com/questions/15012073/opencv-draw-draw-contours-of-2-largest-objects
	int largestIndex = 0;
	int largestContour = 0;
	int secondLargestIndex = 0;
	int thirdLargestIndex = 0;
	int secondLargestContour = 0;
	int thirdLargestContour = 0;
	int fourthLargestContour = 0;
	int fourthLargestIndex = 0;
	int fifthLargestContour = 0;
	int fifthLargestIndex = 0;
	int sixthLargestContour = 0;
	int sixthLargestIndex = 0;

	// Initialize matrices to hold drawings of contours
	Mat drawing1 = Mat::zeros(img.size(), CV_8UC3);
	Mat drawing2 = Mat::zeros(img.size(), CV_8UC3);
	Mat drawing3 = Mat::zeros(img.size(), CV_8UC3);
	Mat drawing4 = Mat::zeros(img.size(), CV_8UC3);
	Mat drawing5 = Mat::zeros(img.size(), CV_8UC3);
	Mat drawing6 = Mat::zeros(img.size(), CV_8UC3);

	// Initialize bounding rectangles that will be used to label the contours 
	Rect bounding_rect1;
	Rect bounding_rect2;
	Rect bounding_rect3;
	Rect bounding_rect4;
	Rect bounding_rect5;
	Rect bounding_rect6;

	// Finds six largest contours
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
			fourthLargestContour = thirdLargestContour;
			fourthLargestIndex = thirdLargestIndex;
			thirdLargestContour = contours[i].size();
			thirdLargestIndex = i;
		}
		else if (contours[i].size() > fourthLargestContour) {
			fifthLargestContour = fourthLargestContour;
			fifthLargestIndex = fourthLargestIndex;
			fourthLargestContour = contours[i].size();
			fourthLargestIndex = i;
		}
		else if (contours[i].size() > fifthLargestContour) {
			sixthLargestContour = fifthLargestContour;
			sixthLargestIndex = fifthLargestIndex;
			fifthLargestContour = contours[i].size();
			fifthLargestIndex = i;
		}
		else if (contours[i].size() > sixthLargestContour) {
			sixthLargestContour = contours[i].size();
			sixthLargestIndex = i;
		}
	}

	bounding_rect1 = boundingRect(contours[largestIndex]);
	bounding_rect2 = boundingRect(contours[secondLargestIndex]);
	bounding_rect3 = boundingRect(contours[thirdLargestIndex]);
	bounding_rect4 = boundingRect(contours[fourthLargestIndex]);
	bounding_rect5 = boundingRect(contours[fifthLargestIndex]);
	bounding_rect6 = boundingRect(contours[sixthLargestIndex]);

	Scalar color = Scalar(0, 0, 255);

	drawContours(drawing1, contours, largestIndex, color, CV_FILLED, 8);
	drawContours(drawing2, contours, secondLargestIndex, color, CV_FILLED, 8);
	drawContours(drawing3, contours, thirdLargestIndex, color, CV_FILLED, 8);
	drawContours(drawing4, contours, fourthLargestIndex, color, CV_FILLED, 8);
	drawContours(drawing5, contours, fifthLargestIndex, color, CV_FILLED, 8);
	drawContours(drawing6, contours, sixthLargestIndex, color, CV_FILLED, 8);

	// Display six largest contours
	imshow("Largest Contour (Source Image)", drawing1);
	imshow("Second Largest Contour (Source Image)", drawing2);
	imshow("Third Largest Contour (Source Image)", drawing3);
	imshow("Fourth Largest Contour (Source Image)", drawing4);
	imshow("Fifth Largest Contour (Source Image)", drawing5);
	imshow("Sixth Largest Contour (Source Image)", drawing6);

	rectangle(labelled_img, bounding_rect1.tl(), bounding_rect1.br(), Scalar(100, 100, 200), 2, CV_AA);
	rectangle(labelled_img, bounding_rect2.tl(), bounding_rect2.br(), Scalar(100, 100, 200), 2, CV_AA);
	rectangle(labelled_img, bounding_rect3.tl(), bounding_rect3.br(), Scalar(100, 100, 200), 2, CV_AA);
	rectangle(labelled_img, bounding_rect4.tl(), bounding_rect4.br(), Scalar(100, 100, 200), 2, CV_AA);
	rectangle(labelled_img, bounding_rect5.tl(), bounding_rect5.br(), Scalar(100, 100, 200), 2, CV_AA);
	rectangle(labelled_img, bounding_rect6.tl(), bounding_rect6.br(), Scalar(100, 100, 200), 2, CV_AA);

	imshow("Labeled Source Image", labelled_img);

	vector<Mat> drawings;
	drawings.push_back(drawing1);
	drawings.push_back(drawing2);
	drawings.push_back(drawing3);
	drawings.push_back(drawing4);
	drawings.push_back(drawing5);
	drawings.push_back(drawing6);

	// Order the contours by their x-coordinates so that the characters are read from left to right
	Rect r[] = { bounding_rect1, bounding_rect2, bounding_rect3, bounding_rect4, bounding_rect5, bounding_rect6 };
	Rect hand1, hand2, hand3, hand4, hand5, hand6;

	float hand1x = float(INT_MAX);
	float hand2x = float(INT_MAX);
	float hand3x = float(INT_MAX);
	float hand4x = float(INT_MAX);
	float hand5x = float(INT_MAX);
	float hand6x = float(INT_MAX);

	Mat d1 = drawing1;
	Mat d2 = drawing2;
	Mat d3 = drawing3;
	Mat d4 = drawing4;
	Mat d5 = drawing5;
	Mat d6 = drawing6;

	for (int i = 0; i < 6; i++)
	{
		// Based on StackOverflow answer here: https://stackoverflow.com/questions/22399257/finding-the-center-of-a-contour-using-opencv-and-visual-c
		float cx = r[i].x + r[i].width / 2;
		if (cx < hand1x) {
			hand6 = hand5;
			hand6x = hand5x;
			d6 = d5;
			hand5 = hand4;
			hand5x = hand4x;
			d5 = d4;
			hand4 = hand3;
			hand4x = hand3x;
			d4 = d3;
			hand3 = hand2;
			hand3x = hand2x;
			d3 = d2;
			hand2 = hand1;
			hand2x = hand1x;
			d2 = d1;
			hand1 = r[i];
			hand1x = cx;
			d1 = drawings[i];
		}
		else if (cx < hand2x) {
			hand6 = hand5;
			hand6x = hand5x;
			d6 = d5;
			hand5 = hand4;
			hand5x = hand4x;
			d5 = d4;
			hand4 = hand3;
			hand4x = hand3x;
			d4 = d3;
			hand3 = hand2;
			hand3x = hand2x;
			d3 = d2;
			hand2 = r[i];
			hand2x = cx;
			d2 = drawings[i];
		}
		else if (cx < hand3x) {
			hand6 = hand5;
			hand6x = hand5x;
			d6 = d5;
			hand5 = hand4;
			hand5x = hand4x;
			d5 = d4;
			hand4 = hand3;
			hand4x = hand3x;
			d4 = d3;
			hand3 = r[i];
			hand3x = cx;
			d3 = drawings[i];
		}
		else if (cx < hand4x) {
			hand6 = hand5;
			hand6x = hand5x;
			d6 = d5;
			hand5 = hand4;
			hand5x = hand4x;
			d5 = d4;
			hand4 = r[i];
			hand4x = cx;
			d4 = drawings[i];
		}
		else if (cx < hand5x) {
			hand6 = hand5;
			hand6x = hand5x;
			d6 = d5;
			hand5 = r[i];
			hand5x = cx;
			d5 = drawings[i];
		}
		else if (cx < hand5x) {
			hand6 = r[i];
			hand6x = cx;
			d6 = drawings[i];
		}
	}

	vector<Mat> ordered_drawings;
	ordered_drawings.push_back(d1);
	ordered_drawings.push_back(d2);
	ordered_drawings.push_back(d3);
	ordered_drawings.push_back(d4);
	ordered_drawings.push_back(d5);
	ordered_drawings.push_back(d6);

	return ordered_drawings;
}

// Finds the largest contour in each template image, draws it on a new image, and returns a grayscale version of the image 
vector<Point> findLargestContour(Mat& img, Mat& labelled_img, Mat& templ) {
	vector<vector<Point>> cont;
	vector<Vec4i> hier;

	// Find contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
	findContours(img, cont, hier, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Convert input image back into a color image so that circles in next part will show up as red 
	cvtColor(img, labelled_img, COLOR_GRAY2BGR);

	// Based on code in StackOverflow answer here: https://stackoverflow.com/questions/15012073/opencv-draw-draw-contours-of-2-largest-objects
	int largestIndex = 0;
	int largestContour = 0;
	Mat drawing1 = Mat::zeros(img.size(), CV_8UC3);
	Rect bounding_rect1;
	for (int i = 0; i < cont.size(); i++)
	{
		if (cont[i].size() > largestContour) {
			largestContour = cont[i].size();
			largestIndex = i;
		}
	}

	bounding_rect1 = boundingRect(cont[largestIndex]);

	Scalar color = Scalar(0, 0, 255);

	drawContours(drawing1, cont, largestIndex, color, CV_FILLED, 8);

	imshow("Largest Contour (Template Image)", drawing1);

	rectangle(labelled_img, bounding_rect1.tl(), bounding_rect1.br(), Scalar(100, 100, 200), 2, CV_AA);

	imshow("Labeled Template Image", labelled_img);

	Mat binaryMat;
	threshold(img, binaryMat, 50, 255, CV_THRESH_BINARY_INV);
	templ = binaryMat;
	return cont[largestIndex];
}

// Function used in first method 
// Based on code from HW2
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

	// imshow("B&W Image", img_bw);
	imshow(" Matching Method - Image Display", img_display);
	imshow("Matching Method - Result", result);

	Rect r(matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows));
	Mat ROI = img_bw(r);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(ROI, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	// Find the largest contour
	int idx = 0;
	double max_area = 0; int max_contour_idx = 0;
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
	//imshow(name, drawing);

	return contours[max_contour_idx];
}

// Function used in second method
// Based on code in comment on this page: https://answers.opencv.org/question/15842/how-to-find-correlation-coefficient-for-two-images/
double correlation(cv::Mat &image1, cv::Mat &image2)
{
	Mat result;
	int match_method = CV_TM_CCORR_NORMED;
	// Other match methods that we experimented with
	//int match_method = CV_TM_CCOEFF_NORMED;
	//int match_method = CV_TM_SQDIFF_NORMED;
	//int match_method = CV_TM_CCORR;

	/// Create the result matrix
	int result_cols = image1.cols - image2.cols + 1;
	int result_rows = image1.rows - image2.rows + 1;

	result.create(result_cols, result_rows, CV_32FC1);

	/// Do the Matching and Normalize
	matchTemplate(image1, image2, result, match_method);
	//normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	return maxVal;
}

// Helper function for MatchingMethod
// Calculates the maximum index of the array
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

// Helper function for MatchingMethod
// Calculates the maximum index of the array
int myMinVal(double* array, int size)
{
	double min = array[0];
	int min_pos = 0;
	int i;
	for (i = 0; i < size; i++)
	{
		if (min > array[i])
		{
			min = array[i];
			min_pos = i;
		}
	}
	return min_pos;
}