/*
* Filename:		ProjectRT.cpp
* Author:		Jun Xu
* Version:		1.0
* Created Time:	Dec. 8, 2014
* Description:	This program
*				(a) extracts and matches SIFT features;
*				(b) computes fundamental and essential matrix
*				(c) computes relative displacement R and T from essential matrix via SVD
*				(d) determines the true value of R uniquely
*				(e) estimates the depth and hence the scene point coordinats
*
* CS585 Video and Image Computing
* Project
* Due Date:		Dec. 8, 2014
*/

#include "stdafx.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"

// number of frames the program will process
#define FRAME_NUM 30
// we compute relative displacement for every 4 frames
#define BASIC_INTERVAL 4

using namespace std;
using namespace cv;

// flag indicating if debug info is shown
int debug = 0;
// flag indicating depth info and estimated scene coordinates (final results) are shown
int showDepth = 0;
// flag indicating matrices including F, E, R, T (intermediate results) are shown
int showTransMat = 0;

// calibration matrix of the camera
double data[] = { 679.1762904713844, 0, 329.4494923815807, 0, 681.184065752318, 201.8327112232685, 0, 0, 1 };
// distortion coefficients matrix of the camera
double dist[] = { -0.2461283687431849, 0.2816828141575686, 0.0002154787123809983, -0.001189677880083738, -0.3734268366833506 };

Mat K(3, 3, CV_64F, data);
Mat D(1, 5, CV_64F, dist);

Mat KInv = K.inv();

// inline function that computes the sign of the input
inline double SIGN(double x) { return (x >= 0.0) ? 1.0 : -1.0; }
// inline function that computes the normal of a 4-element vector
inline double NORM(double a, double b, double c, double d) { return sqrt(a * a + b * b + c * c + d * d); }
// function that computes the relative displacement
int work(int interval, vector< Mat_<double> >& rotationMats, bool tFlag);
// function that computes lambdas from M via SVD
void getLambda(Mat_<double>& lambda, vector<Point2f>& pts1, vector<Point2f>& pts2, Mat_<double>& R, Mat_<double>& T);
// function that returns the element sum of ||R23 * R12| - |R13||
double sumAbsDiff(Mat_<double>& R12, Mat_<double>& R23, Mat_<double>& R13, Mat_<double>& out);

int main(int argc, char** argv)
{
	// vector that store the two solutions of R for every 4 frames
	vector< Mat_<double> > Rs, RTs;
	// vector that store the two solutions of R for every 8 frames
	vector< Mat_<double> > Rs2, RTs2;
	// vector that store the true values of R for every 4 frames
	vector< Mat_<double> > finalRot;
	// ofstream object that is used to store estimated rotation matrices
	ofstream file;

	// compute the two solutions of R for every 4 frames
	work(BASIC_INTERVAL, Rs, false);
	work(BASIC_INTERVAL, RTs, true);
	// compute the two solutions of R for every 8 frames
	work(BASIC_INTERVAL * 2, Rs2, false);
	work(BASIC_INTERVAL * 2, RTs2, true);

	// store the 4 vectors of rotation matrices
	file.open("RotRes4.txt");
	for (int i = 0; i < Rs.size(); i++)
		file << Mat(Rs[i]) << endl << endl;
	file.close();

	file.open("RotTRes4.txt");
	for (int i = 0; i < RTs.size(); i++)
		file << Mat(RTs[i]) << endl << endl;
	file.close();

	file.open("RotRes8.txt");
	for (int i = 0; i < Rs2.size(); i++)
		file << Mat(Rs2[i]) << endl << endl;
	file.close();

	file.open("RotTRes8.txt");
	for (int i = 0; i < RTs2.size(); i++)
		file << Mat(RTs2[i]) << endl << endl;
	file.close();

	// integers that store the indices of R12, R23 and R13 in the above 4 vectors
	int idx12, idx23, idx13;
	// integer that indicates which one of the two possible solutions of R is correct
	int type;

	// for each rotation between every 4 frames, find the true value
	for (int i = 0; i < Rs.size() - 1; i++)
	{
		idx12 = i;
		idx23 = i + 1;
		idx13 = i;

		// double numbers that are used to determine R
		double val = 0.0;
		double minVal = 1000.0;
		// 3-digit hex number, for each digit, 0 means one possible solution of R and 1 means the other
		type = 0x0;

		// double matrix that stores the output value of sumAbsDiff(), which is ||R23 * R12| - |R13||
		Mat_<double> RAbsDiff;
		// for each combination of R12, R23 and R13, if the returned value of sumAbsDiff() is smaller, store it
		val = sumAbsDiff(Rs[idx12], Rs[idx23], Rs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x000;
		}

		val = sumAbsDiff(Rs[idx12], Rs[idx23], RTs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x001;
		}

		val = sumAbsDiff(RTs[idx12], Rs[idx23], Rs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x100;
		}

		val = sumAbsDiff(RTs[idx12], Rs[idx23], RTs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x101;
		}

		val = sumAbsDiff(Rs[idx12], RTs[idx23], Rs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x010;
		}

		val = sumAbsDiff(Rs[idx12], RTs[idx23], RTs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x011;
		}

		val = sumAbsDiff(RTs[idx12], RTs[idx23], Rs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x110;
		}

		val = sumAbsDiff(RTs[idx12], RTs[idx23], RTs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x111;
		}

		// print the 3-digit hex number
		printf("%d:\t0x%3x\n", BASIC_INTERVAL * i, type);

		// store the true value of R
		if (type & 0x10)
			finalRot.push_back(RTs[idx12]);
		else
			finalRot.push_back(Rs[idx12]);
	}

	// store the true value of last R
	if (type & 0x100)
		finalRot.push_back(RTs[idx23]);
	else
		finalRot.push_back(Rs[idx23]);

	// store the true values of consecutive rotations
	file.open("FinalRotRes.txt");
	for (int i = 0; i < finalRot.size(); i++) {
		file << "Frame " << BASIC_INTERVAL * i << " to Frame " << BASIC_INTERVAL * (i + 1) << ":" << endl;
		file << Mat(finalRot[i]) << endl << endl;
	}
	file.close();

	return 0;
}

int work(int interval, vector< Mat_<double> >& rotationMats, bool tFlag)
{
	// stringstream object that will store the file names
	stringstream ss;

	// Mat objects that will store the image
	Mat img1, img2;

	// object that will be used to detect SIFT features in two frames
	SiftFeatureDetector detector(0.05, 5.0);
	// vectors that will store the detected features for each frame
	vector<KeyPoint> keypts1, keypts2;

	// object that will be used to describe the features in two frames
	SiftDescriptorExtractor extractor(3.0);
	// Mat objects that will store the descriptions of features for each frame
	Mat descrs1, descrs2;

	// object that will be used to match two sets of features
	FlannBasedMatcher matcher;
	// vector that will store the matching info
	vector<DMatch> matches;

	// double numbers that will store the maximum and minimum distances between each pair of matching features
	double maxDist, minDist;
	// vector that will store the matching info after a simple filter
	vector<DMatch> goodMatches;
	// point vectors that will store the image coordinates after a simple filter for each of the two frames
	vector<Point2f> pt1, pt2;

	// vector that will indicate whether a point is an inlier in RANSAC
	vector<uchar> masks;
	// vector that will store the matching info between RANSAC inliers
	vector<DMatch> betterMatches;
	// vector that will store the image coordinates for all RANSAC inliers
	vector<Point2f> pts1, pts2;

	// Mat object that will store the combined images and the matching lines
	Mat imgMatches2;

	// Mat object that will store the fundamental matrix
	Mat F;
	// Mat object that will store the essential matrix
	Mat_<double> E;
	// Mat objects that will store the U, W, transposed V while computing R and T
	Mat svd_u, svd_vt, svd_w;

	// W matrix
	Matx33d W(0, -1, 0, 1, 0, 0, 0, 0, 1);
	// Mat objects that will store the info of R and T
	Mat_<double> R, T;

	// Mat object that will store the estimates lambdas
	Mat_<double> lambda;
	// Mat object that will store the value of K (calibration matrix) inverse
	Mat KInv;
	// Mat object that will store the estimated scene coordinates
	vector<Point3f> scenePts;

	// for every BASIC_INTERVAL frames, compute the relative displacement
	for (int i = 0; i < FRAME_NUM; i += BASIC_INTERVAL)
	{
		// create two file names
		ss.str("");
		ss << 40 + i;
		String filename1 = "2014-12-02_Office_Scene/image_00" + ss.str() + ".jpg";
		ss.str("");
		ss << 40 + i + interval;
		String filename2 = "2014-12-02_Office_Scene/image_00" + ss.str() + ".jpg";

		// read the two images
		img1 = imread(filename1, CV_LOAD_IMAGE_UNCHANGED);
		img2 = imread(filename2, CV_LOAD_IMAGE_UNCHANGED);
		if (!img1.data || !img2.data)
		{
			cout << "Error reading images" << endl;
			return -1;
		}

		// undistort the two images based on camera distortion coefficients
		undistort(img1.clone(), img1, K, D);
		undistort(img2.clone(), img2, K, D);

		// detect the SIFT features
		keypts1.clear();
		keypts2.clear();
		detector.detect(img1, keypts1);
		detector.detect(img2, keypts2);

		// describe the SIFT features
		descrs1.release();
		descrs2.release();
		extractor.compute(img1, keypts1, descrs1);
		extractor.compute(img2, keypts2, descrs2);
		if (debug)
		{
			cout << "Image 1:\t" << keypts1.size() << " keypoints" << endl;
			cout << "Image 2:\t" << keypts2.size() << " keypoints\n" << endl;
		}

		// match two sets of features
		matches.clear();
		matcher.match(descrs1, descrs2, matches);

		// compute the min and max distance of matched features
		maxDist = 0;
		minDist = 100;
		for (int i = 0; i < descrs1.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < minDist) minDist = dist;
			if (dist > maxDist) maxDist = dist;
		}
		if (debug)
		{
			cout << "Max distance:\t" << maxDist << endl;
			cout << "Min distance:\t" << minDist << "\n" << endl;
		}

		// keep a pair of features if their distance of is smaller than 3 * min distance
		goodMatches.clear();
		for (int i = 0; i < descrs1.rows; i++)
		{
			if (matches[i].distance < 3 * minDist)
				goodMatches.push_back(matches[i]);
		}

		// find the features after above the simple filter
		pt1.clear();
		pt2.clear();
		for (int i = 0; i < goodMatches.size(); i++)
		{
			pt1.push_back(keypts1[goodMatches[i].queryIdx].pt);
			pt2.push_back(keypts2[goodMatches[i].trainIdx].pt);
		}

		// find the fundamental matrix from two sets of features
		masks.clear();
		F.release();
		F = findFundamentalMat(pt1, pt2, CV_RANSAC, 3, 0.999, masks);
		if (showTransMat)
		{
			cout << "Fundamental Matrix\nRow:\t" << F.rows << endl;
			cout << "Col:\t" << F.cols << endl;
			cout << F << endl;
		}

		// find the features as RANSAC inliers
		betterMatches.clear();
		pts1.clear();
		pts2.clear();
		for (int i = 0; i < goodMatches.size(); i++)
		{
			if (masks[i]) {
				betterMatches.push_back(goodMatches[i]);
				pts1.push_back(keypts1[goodMatches[i].queryIdx].pt);
				pts2.push_back(keypts2[goodMatches[i].trainIdx].pt);
				if (debug)
				{
					cout << "(x, y): ("
						<< keypts1[goodMatches[i].queryIdx].pt.x << ", "
						<< keypts1[goodMatches[i].queryIdx].pt.y << ")\t("
						<< keypts2[goodMatches[i].trainIdx].pt.x << ", "
						<< keypts2[goodMatches[i].trainIdx].pt.y << ")" << endl;
				}
			}
		}
		if (debug) cout << endl;

		// draw the combined images and matching lines
		imgMatches2.release();
		drawMatches(img1, keypts1, img2, keypts2, betterMatches, imgMatches2, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//imshow( "Good Matches & Object detection 2", imgMatches2 );

		// compute E from F
		E.release();
		E = K.t() * F * K;
		if (showTransMat)
		{
			cout << "\nEssential Matrix\nRow:\t" << E.rows << endl;
			cout << "Col:\t" << E.cols << endl;
			cout << E << endl;
		}

		// compute R and T from E via SVD
		SVD svd(E, SVD::MODIFY_A);
		svd_u.release();
		svd_vt.release();
		svd_w.release();
		svd_u = svd.u;
		svd_vt = svd.vt;
		svd_w = svd.w;

		R.release();
		T.release();
		if (tFlag)
			R = -svd_u * Mat(W).t() * svd_vt;
		else
			R = -svd_u * Mat(W) * svd_vt;
		T = svd_u.col(2);
		if (showTransMat)
		{
			cout << "\nR\nRow:\t" << R.rows << endl;
			cout << "Col:\t" << R.cols << endl;
			cout << R << endl;
			cout << "\nT\nRow:\t" << T.rows << endl;
			cout << "Col:\t" << T.cols << endl;
			cout << T << endl;
		}
		// store the estimated R to a vector
		rotationMats.push_back(R);

		// compute the lambdas 
		lambda.release();
		getLambda(lambda, pts1, pts2, R, T);
		if (showDepth)
		{
			cout << "\nLambda\nRow:\t" << lambda.rows << endl;
			cout << "Col:\t" << lambda.cols << endl;
			cout << Mat(lambda) << endl;
		}

		if (debug)
		{
			cout << "\n1st Image Points" << endl;
			cout << Mat(pts1) << endl;

			cout << "\n2nd Image Points" << endl;
			cout << Mat(pts2) << endl;
		}

		if (debug)
		{
			cout << "\nK Inverse\nRow:\t" << KInv.rows << endl;
			cout << "Col:\t" << KInv.cols << endl;
			cout << K.inv() << endl;
		}

		// compute the estimated scene point coordinates
		scenePts.clear();
		for (int i = 0; i < pts1.size(); i++)
		{
			Mat_<double> p = (Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1);

			Mat_<double> KInvp = K.inv() * p;
			double l = -lambda.at<double>(i, 0);
			Point3d scenePt(KInvp.at<double>(0, 0) * l, KInvp.at<double>(1, 0) * l, KInvp.at<double>(2, 0) * l);
			scenePts.push_back(scenePt);
		}
		if (showDepth)
		{
			cout << "\nScene Points Coordinates" << endl;
			cout << Mat(scenePts) << endl;
		}

		// compute the true translation vector
		for (int i = 0; i < T.rows; i++)
			T.at<double>(i, 0) *= lambda.at<double>(lambda.rows - 1, 0);
		if (showTransMat)
		{
			cout << "\nReal T\nRow:\t" << T.rows << endl;
			cout << "Col:\t" << T.cols << endl;
			cout << T << endl;
		}

		//cv::waitKey(0);
	}

	return 0;
}

void getLambda(Mat_<double>& lambda, vector<Point2f>& pts1, vector<Point2f>& pts2, Mat_<double>& R, Mat_<double>& T)
{
	// compute the number of rows and columns
	int rows = 3 * pts1.size();
	int cols = pts1.size() + 1;
	// Mat object that will store the value of M, refer to report for explanation of M
	Mat M = Mat::zeros(rows, cols, CV_64FC1);

	// construct the M
	for (int i = 0; i < pts1.size(); i++)
	{
		Mat x1 = (Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
		Mat x2 = (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);

		x1 = KInv * x1;
		x2 = KInv * x2;

		Mat_<double> xRx = x2.cross(R * x1);
		M.at<double>(3 * i, i) = xRx.at<double>(0, 0);
		M.at<double>(3 * i + 1, i) = xRx.at<double>(1, 0);
		M.at<double>(3 * i + 2, i) = xRx.at<double>(2, 0);

		Mat_<double> xT = x2.cross(T);
		M.at<double>(3 * i, cols - 1) = xT.at<double>(0, 0);
		M.at<double>(3 * i + 1, cols - 1) = xT.at<double>(1, 0);
		M.at<double>(3 * i + 2, cols - 1) = xT.at<double>(2, 0);
	}

	// compute lambdas from M via SVD
	SVD svd(M, SVD::MODIFY_A);
	Mat svd_u = svd.u;
	Mat svd_vt = svd.vt;
	Mat svd_v = svd_vt.t();
	Mat svd_w = svd.w;

	lambda = svd_v.col(cols - 1);
}

double sumAbsDiff(Mat_<double>& R12, Mat_<double>& R23, Mat_<double>& R13, Mat_<double>& out)
{
	// 3d double array that will store R23 * R12 and R13
	double R[2][3][3];
	// double number that will store the absolute difference between |R23 * R12| and |R13|
	double result = 0.0;

	// compute R23 * R12
	Mat_<double> R123 = R23 * R12;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				if (i == 0)
					R[i][j][k] = R123.at<double>(j, k);
				else
					R[i][j][k] = R13.at<double>(j, k);
			}
		}
	}

	// compute the absolute diff between |R23 * R12| and |R13|
	out = Mat(3, 3, CV_64F);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			double val = abs(abs(R[0][i][j]) - abs(R[1][i][j]));
			out.at<double>(i, j) = val;
			result += val;
		}
	}

	return result;
}/*
* Filename:		ProjectRT.cpp
* Author:		Jun Xu
* Version:		1.0
* Created Time:	Dec. 8, 2014
* Description:	This program 
*				(a) extracts and matches SIFT features;
*				(b) computes fundamental and essential matrix
*				(c) computes relative displacement R and T from essential matrix via SVD
*				(d) determines the true value of R uniquely
*				(e) estimates the depth and hence the scene point coordinats
*
* CS585 Video and Image Computing
* Project
* Due Date:		Dec. 8, 2014
*/

#include "stdafx.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"

// number of frames the program will process
#define FRAME_NUM 30
// we compute relative displacement for every 4 frames
#define BASIC_INTERVAL 4

using namespace std;
using namespace cv;

// flag indicating if debug info is shown
int debug = 0;
// flag indicating depth info and estimated scene coordinates (final results) are shown
int showDepth = 0;
// flag indicating matrices including F, E, R, T (intermediate results) are shown
int showTransMat = 0;

// calibration matrix of the camera
double data[] = { 679.1762904713844, 0, 329.4494923815807, 0, 681.184065752318, 201.8327112232685, 0, 0, 1 };
// distortion coefficients matrix of the camera
double dist[] = { -0.2461283687431849, 0.2816828141575686, 0.0002154787123809983, -0.001189677880083738, -0.3734268366833506 };

Mat K(3, 3, CV_64F, data);
Mat D(1, 5, CV_64F, dist);

Mat KInv = K.inv();

// inline function that computes the sign of the input
inline double SIGN(double x) { return (x >= 0.0) ? 1.0 : -1.0; }
// inline function that computes the normal of a 4-element vector
inline double NORM(double a, double b, double c, double d) { return sqrt(a * a + b * b + c * c + d * d); }
// function that computes the relative displacement
int work(int interval, vector< Mat_<double> >& rotationMats, bool tFlag);
// function that computes lambdas from M via SVD
void getLambda(Mat_<double>& lambda, vector<Point2f>& pts1, vector<Point2f>& pts2, Mat_<double>& R, Mat_<double>& T);
// function that returns the element sum of ||R23 * R12| - |R13||
double sumAbsDiff(Mat_<double>& R12, Mat_<double>& R23, Mat_<double>& R13, Mat_<double>& out);

int main(int argc, char** argv)
{
	// vector that store the two solutions of R for every 4 frames
	vector< Mat_<double> > Rs, RTs;
	// vector that store the two solutions of R for every 8 frames
	vector< Mat_<double> > Rs2, RTs2;
	// vector that store the true values of R for every 4 frames
	vector< Mat_<double> > finalRot;
	// ofstream object that is used to store estimated rotation matrices
	ofstream file;

	// compute the two solutions of R for every 4 frames
	work(BASIC_INTERVAL, Rs, false);
	work(BASIC_INTERVAL, RTs, true);
	// compute the two solutions of R for every 8 frames
	work(BASIC_INTERVAL * 2, Rs2, false);
	work(BASIC_INTERVAL * 2, RTs2, true);

	// store the 4 vectors of rotation matrices
	file.open("RotRes4.txt");
	for (int i = 0; i < Rs.size(); i++)
		file << Mat(Rs[i]) << endl << endl;
	file.close();

	file.open("RotTRes4.txt");
	for (int i = 0; i < RTs.size(); i++)
		file << Mat(RTs[i]) << endl << endl;
	file.close();

	file.open("RotRes8.txt");
	for (int i = 0; i < Rs2.size(); i++)
		file << Mat(Rs2[i]) << endl << endl;
	file.close();

	file.open("RotTRes8.txt");
	for (int i = 0; i < RTs2.size(); i++)
		file << Mat(RTs2[i]) << endl << endl;
	file.close();

	// integers that store the indices of R12, R23 and R13 in the above 4 vectors
	int idx12, idx23, idx13;
	// integer that indicates which one of the two possible solutions of R is correct
	int type;

	// for each rotation between every 4 frames, find the true value
	for (int i = 0; i < Rs.size() - 1; i++)
	{
		idx12 = i;
		idx23 = i + 1;
		idx13 = i;

		// double numbers that are used to determine R
		double val = 0.0;
		double minVal = 1000.0;
		// 3-digit hex number, for each digit, 0 means one possible solution of R and 1 means the other
		type = 0x0;

		// double matrix that stores the output value of sumAbsDiff(), which is ||R23 * R12| - |R13||
		Mat_<double> RAbsDiff;
		// for each combination of R12, R23 and R13, if the returned value of sumAbsDiff() is smaller, store it
		val = sumAbsDiff(Rs[idx12], Rs[idx23], Rs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x000;
		}

		val = sumAbsDiff(Rs[idx12], Rs[idx23], RTs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x001;
		}

		val = sumAbsDiff(RTs[idx12], Rs[idx23], Rs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x100;
		}

		val = sumAbsDiff(RTs[idx12], Rs[idx23], RTs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x101;
		}

		val = sumAbsDiff(Rs[idx12], RTs[idx23], Rs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x010;
		}

		val = sumAbsDiff(Rs[idx12], RTs[idx23], RTs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x011;
		}

		val = sumAbsDiff(RTs[idx12], RTs[idx23], Rs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x110;
		}

		val = sumAbsDiff(RTs[idx12], RTs[idx23], RTs2[idx13], RAbsDiff);
		if (val < minVal)
		{
			minVal = val;
			type = 0x111;
		}

		// print the 3-digit hex number
		printf("%d:\t0x%3x\n", BASIC_INTERVAL * i, type);

		// store the true value of R
		if (type & 0x10)
			finalRot.push_back(RTs[idx12]);
		else
			finalRot.push_back(Rs[idx12]);
	}

	// store the true value of last R
	if (type & 0x100)
		finalRot.push_back(RTs[idx23]);
	else
		finalRot.push_back(Rs[idx23]);

	// store the true values of consecutive rotations
	file.open("FinalRotRes.txt");
	for (int i = 0; i < finalRot.size(); i++) {
		file << "Frame " << BASIC_INTERVAL * i << " to Frame " << BASIC_INTERVAL * (i + 1) << ":" << endl;
		file << Mat(finalRot[i]) << endl << endl;
	}
	file.close();

	return 0;
}

int work(int interval, vector< Mat_<double> >& rotationMats, bool tFlag)
{
	// stringstream object that will store the file names
	stringstream ss;

	// Mat objects that will store the image
	Mat img1, img2;

	// object that will be used to detect SIFT features in two frames
	SiftFeatureDetector detector(0.05, 5.0);
	// vectors that will store the detected features for each frame
	vector<KeyPoint> keypts1, keypts2;

	// object that will be used to describe the features in two frames
	SiftDescriptorExtractor extractor(3.0);
	// Mat objects that will store the descriptions of features for each frame
	Mat descrs1, descrs2;

	// object that will be used to match two sets of features
	FlannBasedMatcher matcher;
	// vector that will store the matching info
	vector<DMatch> matches;

	// double numbers that will store the maximum and minimum distances between each pair of matching features
	double maxDist, minDist;
	// vector that will store the matching info after a simple filter
	vector<DMatch> goodMatches;
	// point vectors that will store the image coordinates after a simple filter for each of the two frames
	vector<Point2f> pt1, pt2;

	// vector that will indicate whether a point is an inlier in RANSAC
	vector<uchar> masks;
	// vector that will store the matching info between RANSAC inliers
	vector<DMatch> betterMatches;
	// vector that will store the image coordinates for all RANSAC inliers
	vector<Point2f> pts1, pts2;

	// Mat object that will store the combined images and the matching lines
	Mat imgMatches2;

	// Mat object that will store the fundamental matrix
	Mat F;
	// Mat object that will store the essential matrix
	Mat_<double> E;
	// Mat objects that will store the U, W, transposed V while computing R and T
	Mat svd_u, svd_vt, svd_w;

	// W matrix
	Matx33d W(0, -1, 0, 1, 0, 0, 0, 0, 1);
	// Mat objects that will store the info of R and T
	Mat_<double> R, T;

	// Mat object that will store the estimates lambdas
	Mat_<double> lambda;
	// Mat object that will store the value of K (calibration matrix) inverse
	Mat KInv;
	// Mat object that will store the estimated scene coordinates
	vector<Point3f> scenePts;

	// for every BASIC_INTERVAL frames, compute the relative displacement
	for (int i = 0; i < FRAME_NUM; i += BASIC_INTERVAL)
	{
		// create two file names
		ss.str("");
		ss << 40 + i;
		String filename1 = "2014-12-02_Office_Scene/image_00" + ss.str() + ".jpg";
		ss.str("");
		ss << 40 + i + interval;
		String filename2 = "2014-12-02_Office_Scene/image_00" + ss.str() + ".jpg";

		// read the two images
		img1 = imread(filename1, CV_LOAD_IMAGE_UNCHANGED);
		img2 = imread(filename2, CV_LOAD_IMAGE_UNCHANGED);
		if (!img1.data || !img2.data)
		{
			cout << "Error reading images" << endl;
			return -1;
		}

		// undistort the two images based on camera distortion coefficients
		undistort(img1.clone(), img1, K, D);
		undistort(img2.clone(), img2, K, D);

		// detect the SIFT features
		keypts1.clear();
		keypts2.clear();
		detector.detect(img1, keypts1);
		detector.detect(img2, keypts2);

		// describe the SIFT features
		descrs1.release();
		descrs2.release();
		extractor.compute(img1, keypts1, descrs1);
		extractor.compute(img2, keypts2, descrs2);
		if (debug)
		{
			cout << "Image 1:\t" << keypts1.size() << " keypoints" << endl;
			cout << "Image 2:\t" << keypts2.size() << " keypoints\n" << endl;
		}

		// match two sets of features
		matches.clear();
		matcher.match(descrs1, descrs2, matches);

		// compute the min and max distance of matched features
		maxDist = 0;
		minDist = 100;
		for (int i = 0; i < descrs1.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < minDist) minDist = dist;
			if (dist > maxDist) maxDist = dist;
		}
		if (debug)
		{
			cout << "Max distance:\t" << maxDist << endl;
			cout << "Min distance:\t" << minDist << "\n" << endl;
		}

		// keep a pair of features if their distance of is smaller than 3 * min distance
		goodMatches.clear();
		for (int i = 0; i < descrs1.rows; i++)
		{
			if (matches[i].distance < 3 * minDist)
				goodMatches.push_back(matches[i]);
		}

		// find the features after above the simple filter
		pt1.clear();
		pt2.clear();
		for (int i = 0; i < goodMatches.size(); i++)
		{
			pt1.push_back(keypts1[goodMatches[i].queryIdx].pt);
			pt2.push_back(keypts2[goodMatches[i].trainIdx].pt);
		}

		// find the fundamental matrix from two sets of features
		masks.clear();
		F.release();
		F = findFundamentalMat(pt1, pt2, CV_RANSAC, 3, 0.999, masks);
		if (showTransMat)
		{
			cout << "Fundamental Matrix\nRow:\t" << F.rows << endl;
			cout << "Col:\t" << F.cols << endl;
			cout << F << endl;
		}

		// find the features as RANSAC inliers
		betterMatches.clear();
		pts1.clear();
		pts2.clear();
		for (int i = 0; i < goodMatches.size(); i++)
		{
			if (masks[i]) {
				betterMatches.push_back(goodMatches[i]);
				pts1.push_back(keypts1[goodMatches[i].queryIdx].pt);
				pts2.push_back(keypts2[goodMatches[i].trainIdx].pt);
				if (debug)
				{
					cout << "(x, y): ("
						<< keypts1[goodMatches[i].queryIdx].pt.x << ", "
						<< keypts1[goodMatches[i].queryIdx].pt.y << ")\t("
						<< keypts2[goodMatches[i].trainIdx].pt.x << ", "
						<< keypts2[goodMatches[i].trainIdx].pt.y << ")" << endl;
				}
			}
		}
		if (debug) cout << endl;

		// draw the combined images and matching lines
		imgMatches2.release();
		drawMatches(img1, keypts1, img2, keypts2, betterMatches, imgMatches2, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//imshow( "Good Matches & Object detection 2", imgMatches2 );

		// compute E from F
		E.release();
		E = K.t() * F * K;
		if (showTransMat)
		{
			cout << "\nEssential Matrix\nRow:\t" << E.rows << endl;
			cout << "Col:\t" << E.cols << endl;
			cout << E << endl;
		}

		// compute R and T from E via SVD
		SVD svd(E, SVD::MODIFY_A);
		svd_u.release();
		svd_vt.release();
		svd_w.release();
		svd_u = svd.u;
		svd_vt = svd.vt;
		svd_w = svd.w;

		R.release();
		T.release();
		if (tFlag)
			R = -svd_u * Mat(W).t() * svd_vt;
		else
			R = -svd_u * Mat(W) * svd_vt;
		T = svd_u.col(2);
		if (showTransMat)
		{
			cout << "\nR\nRow:\t" << R.rows << endl;
			cout << "Col:\t" << R.cols << endl;
			cout << R << endl;
			cout << "\nT\nRow:\t" << T.rows << endl;
			cout << "Col:\t" << T.cols << endl;
			cout << T << endl;
		}
		// store the estimated R to a vector
		rotationMats.push_back(R);

		// compute the lambdas 
		lambda.release();
		getLambda(lambda, pts1, pts2, R, T);
		if (showDepth)
		{
			cout << "\nLambda\nRow:\t" << lambda.rows << endl;
			cout << "Col:\t" << lambda.cols << endl;
			cout << Mat(lambda) << endl;
		}

		if (debug)
		{
			cout << "\n1st Image Points" << endl;
			cout << Mat(pts1) << endl;

			cout << "\n2nd Image Points" << endl;
			cout << Mat(pts2) << endl;
		}

		if (debug)
		{
			cout << "\nK Inverse\nRow:\t" << KInv.rows << endl;
			cout << "Col:\t" << KInv.cols << endl;
			cout << K.inv() << endl;
		}

		// compute the estimated scene point coordinates
		scenePts.clear();
		for (int i = 0; i < pts1.size(); i++)
		{
			Mat_<double> p = (Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1);

			Mat_<double> KInvp = K.inv() * p;
			double l = -lambda.at<double>(i, 0);
			Point3d scenePt(KInvp.at<double>(0, 0) * l, KInvp.at<double>(1, 0) * l, KInvp.at<double>(2, 0) * l);
			scenePts.push_back(scenePt);
		}
		if (showDepth)
		{
			cout << "\nScene Points Coordinates" << endl;
			cout << Mat(scenePts) << endl;
		}

		// compute the true translation vector
		for (int i = 0; i < T.rows; i++)
			T.at<double>(i, 0) *= lambda.at<double>(lambda.rows - 1, 0);
		if (showTransMat)
		{
			cout << "\nReal T\nRow:\t" << T.rows << endl;
			cout << "Col:\t" << T.cols << endl;
			cout << T << endl;
		}

		//cv::waitKey(0);
	}

	return 0;
}

void getLambda(Mat_<double>& lambda, vector<Point2f>& pts1, vector<Point2f>& pts2, Mat_<double>& R, Mat_<double>& T)
{
	// compute the number of rows and columns
	int rows = 3 * pts1.size();
	int cols = pts1.size() + 1;
	// Mat object that will store the value of M, refer to report for explanation of M
	Mat M = Mat::zeros(rows, cols, CV_64FC1);

	// construct the M
	for (int i = 0; i < pts1.size(); i++)
	{
		Mat x1 = (Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
		Mat x2 = (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);

		x1 = KInv * x1;
		x2 = KInv * x2;

		Mat_<double> xRx = x2.cross(R * x1);
		M.at<double>(3 * i, i) = xRx.at<double>(0, 0);
		M.at<double>(3 * i + 1, i) = xRx.at<double>(1, 0);
		M.at<double>(3 * i + 2, i) = xRx.at<double>(2, 0);

		Mat_<double> xT = x2.cross(T);
		M.at<double>(3 * i, cols - 1) = xT.at<double>(0, 0);
		M.at<double>(3 * i + 1, cols - 1) = xT.at<double>(1, 0);
		M.at<double>(3 * i + 2, cols - 1) = xT.at<double>(2, 0);
	}

	// compute lambdas from M via SVD
	SVD svd(M, SVD::MODIFY_A);
	Mat svd_u = svd.u;
	Mat svd_vt = svd.vt;
	Mat svd_v = svd_vt.t();
	Mat svd_w = svd.w;

	lambda = svd_v.col(cols - 1);
}

double sumAbsDiff(Mat_<double>& R12, Mat_<double>& R23, Mat_<double>& R13, Mat_<double>& out)
{
	// 3d double array that will store R23 * R12 and R13
	double R[2][3][3];
	// double number that will store the absolute difference between |R23 * R12| and |R13|
	double result = 0.0;

	// compute R23 * R12
	Mat_<double> R123 = R23 * R12;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				if (i == 0)
					R[i][j][k] = R123.at<double>(j, k);
				else
					R[i][j][k] = R13.at<double>(j, k);
			}
		}
	}

	// compute the absolute diff between |R23 * R12| and |R13|
	out = Mat(3, 3, CV_64F);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			double val = abs(abs(R[0][i][j]) - abs(R[1][i][j]));
			out.at<double>(i, j) = val;
			result += val;
		}
	}

	return result;
}