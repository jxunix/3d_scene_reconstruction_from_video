/*--------------------------------------------------------------------------
 
 File Name:         main.cpp
 Date Created:      2014/12/03
 Date Modified:     2014/12/03
 
 Author:            Eric Cristofalo
 Contact:           eric.cristofalo@gmail.com
 
 Description:       Camera Calibration
                    This program calibrates any video stream (currently set up as the built in webcam on a laptop). Once run, you are asked to take the number of pictures you would like to use for the calibration (advised n>=10) and the video stream will be displayed on the screen allowing the user to position a camera calibration pattern in the view. Once recognized, the calibration points will be displayed on the screen. Press space bar to capture a particular image until the n images are taken. The camera calibration matrix and distortion coefficients will be output afterwards. 
 
 -------------------------------------------------------------------------*/

// Include Libraries
#include <iostream>                             // C++ Standard Input/Output
#include <stdio.h>
#include <cmath>                                // C++ Math Library
#include <vector>
#include <unistd.h>                             // Unix Standard Library
#include "opencv2/opencv.hpp"                   // OpenCV Library
#include "opencv2/core/core.hpp"                // OpenCV core library
#include "opencv2/highgui/highgui.hpp"          // OpenCV highgui library
#include "opencv2/imgproc/imgproc.hpp"          // OpenCV image processing library
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"

// Namespaces
using namespace cv;
using namespace std;

// Global Variables
Mat image;

int main(int argc, char *argv[])
{
    // Initialize Built-in Webcam
    VideoCapture cap(0); // open the video camera no. 0
    if (!cap.isOpened()) // if not successful, exit program
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }
    
    // Acquire Frame
    bool bSuccess0 = cap.read(image);
    if (!bSuccess0) //if not successful, break loop
    {
        cout << "Cannot read a frame from video stream" << endl;
    }
    
    // Initialization
    int nTrials;             // number of images of chessboard
    cout << "\n\n-------------------- Please Enter an Input --------------------\n";
    cout << "Enter the number of images you will take for calibration: \n";
    cin >> nTrials;
    int nCornersHor = 8;    // number of corners in horizontal direction
    int nCornersVer = 6;    // number of corners in vertical direction
    int numSquares = nCornersHor * nCornersVer;
    
    // Images
    Size sizeBoard = Size(nCornersHor, nCornersVer);
    Mat imageGray, imageFlip;
    
    // Vectors of Vectors
    vector< vector< Point3f > > objectPoints;   // points in 3D space
    vector< vector< Point2f > > imagePoints;    // points in image
    vector< Point3f > obj;                      // corners coordinates
    vector< Point2f > corners;                  // corners
    int successes=0;
    
    // Set Up Calibration Corners
    for (int i=0;i<numSquares;i++) {
        obj.push_back(Point3f(i/nCornersHor, i%nCornersHor, 0.0f));
        //cout << "obj:\n" << obj << "\n";
    }

    // Keep Taking Calibration Images
    cout << "Press the space bar to take an image\n";
    while (successes<=(nTrials+1)) {
        
        // Read a New Frame from Video Stream
        bool bSuccess = cap.read(image);
        if (!bSuccess) //if not successful, break loop
        {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }
        if (waitKey(1) == 27) break; // NECESSARY TO DISPLAY FIGURES
        
        // Find Chessboard Corners
        cvtColor(image, imageGray, CV_BGR2GRAY); // gray scale;
        bool found = findChessboardCorners(image, sizeBoard, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

        if (found) {
            cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(image, sizeBoard, corners, found);
        }
        flip(image, imageFlip, 1);
        imshow("Calibration Image",imageFlip);
        
        // Check to Store Image
        int key = waitKey(1);
        //cout << "Pressed Key: " << key << endl;
        if (key==27) { // ESC quits the program
            return 0;
        }
        else if (key==32 && found!=0) {  // space bar saves the image for the calibration
            imagePoints.push_back(corners);
            objectPoints.push_back(obj);
            successes++;
            cout << "snap " << successes << " of " << nTrials << " saved!\n";
            if(successes >= nTrials) break;
        } // end waiting for keystroke if
    } // end while
    cvDestroyWindow("Calibration Image");
    
    // Initialize Camera Calibration
    Mat cameraCalibration = Mat(3, 3, CV_32FC1);
    cameraCalibration.ptr<float>(0)[0] = 1;
    cameraCalibration.ptr<float>(1)[1] = 1;
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    
    // Perform Camera Calibration
    calibrateCamera(objectPoints, imagePoints, image.size(), cameraCalibration, distCoeffs, rvecs, tvecs);
    cout << "------------------------------------------------------------\n";
    cout << "Camera Calibration Matrix: \n" << cameraCalibration << "\n\n";
    cout << "Camera Distortion Coefficients: \n" << distCoeffs << "\n\n";
    cout << "------------------------------------------------------------\n";
    
    // Run Undistorted Demo From Calibration
    Mat image = imread("doNotDelete.jpg",3);
    Mat imageUndistorted;
    namedWindow("Raw Image", WINDOW_AUTOSIZE);
    namedWindow("Calibrated Image", WINDOW_AUTOSIZE);
    
    while(1) {
        // Read a New Frame from Video Stream
        bool bSuccess = cap.read(image);
        if (!bSuccess) //if not successful, break loop
        {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }
        if (waitKey(1) == 27) break; // NECESSARY TO DISPLAY FIGURES
        flip(image, imageFlip, 1);

        // Undistort Image and Compare
        undistort(imageFlip, imageUndistorted, cameraCalibration, distCoeffs);
        imshow("Raw Image", imageFlip);
        imshow("Calibrated Image", imageUndistorted);
    }
    
    return 0;
}

