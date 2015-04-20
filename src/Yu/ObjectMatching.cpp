//
//  ObjectMatching.cpp
//  CS585_final_project
//
//  Created by Chen Yu on 11/8/14.
//  Copyright (c) 2014 Chen Yu. All rights reserved.
//

#include "ObjectMatching.h"

#include "opencv2/imgproc/imgproc.hpp"

int matching(string path1, string path2, Mat &H,
             vector<KeyPoint>& keypts1, vector<KeyPoint>& keypts2,
            Mat& descrs1, Mat& descrs2,
             vector<DMatch>& matches){
    
    Mat img1 = imread(path1, CV_LOAD_IMAGE_UNCHANGED);
    Mat img2 = imread(path2, CV_LOAD_IMAGE_UNCHANGED);
    
    double Cam_Mat[] = {679.1762904713844, 0, 329.4494923815807,
        0, 681.184065752318, 201.8327112232685,
        0, 0,1};
    
    Mat Cam_cal(3, 3, CV_64F, Cam_Mat);
    
    
    double Distort_data[] = {
        -0.2461283687431849,
        0.2816828141575686,
        0.0002154787123809983,
        -0.001189677880083738,
        -0.3734268366833506};
    
     Mat Distort_Mat(1, 5, CV_64F, Distort_data);
    
    undistort(img1.clone(), img1, Cam_cal, Distort_Mat);
    undistort(img2.clone(), img2, Cam_cal, Distort_Mat);
    
    if (!img1.data || !img2.data)
    {
        cout << "Error reading images" << endl;
        return -1;
    }
    
    // Step 1: Detect the keypoints using SIFT Detector
#ifdef SIFT
    SiftFeatureDetector detector(0.1,1.0);
#endif
#ifdef SURF
    SurfFeatureDetector detector(400);
#endif
//    vector<KeyPoint> keypts1, keypts2;
    detector.detect(img1, keypts1);
    detector.detect(img2, keypts2);
    
    // Step 2: Calculate descriptors (feature vectors)
#ifdef SIFT
    SiftDescriptorExtractor extractor(3.0);
#endif
#ifdef SURF
    SurfDescriptorExtractor extractor;
#endif
//    Mat descrs1, descrs2;
    extractor.compute(img1, keypts1, descrs1);
    extractor.compute(img2, keypts2, descrs2);
    
//    cout << "Image 1:\t" << keypts1.size() << " keypoints" << endl;
//    cout << "Image 2:\t" << keypts2.size() << " keypoints\n" << endl;
    
    // Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
//    vector<DMatch> matches;
    matcher.match(descrs1, descrs2, matches);
    
    // Quick calculation of max and min distances between keypoints
    double maxDist = 0;
    double minDist = 100;
    for (int i = 0; i < descrs1.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < minDist) minDist = dist;
        if (dist > maxDist) maxDist = dist;
    }
    
    // Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    vector<DMatch> goodMatches;
    for (int i = 0; i < descrs1.rows; i++)
    {
        if (matches[i].distance < 7 * minDist)
        {
            goodMatches.push_back(matches[i]);
        }
    }
    
//    Mat imgMatches;
//    drawMatches(img1, keypts1, img2, keypts2, goodMatches, imgMatches, Scalar::all(-1), Scalar::all(-1),
//                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    
    // Localize the object
    vector<Point2f> pt1, pt2;
    for (int i = 0; i < goodMatches.size(); i++)
    {
        // Get the keypoints from the good matches
        pt1.push_back(keypts1[goodMatches[i].queryIdx].pt);
        pt2.push_back(keypts2[goodMatches[i].trainIdx].pt);
    }
    
//    cout << "TEST:"<< goodMatches[1].imgIdx<<endl;
//    cout << "TEST:"<< goodMatches[1].distance <<endl;
//    cout << "TEST:"<< goodMatches[1].queryIdx <<endl;
//    cout << "TEST:"<< goodMatches[1].trainIdx <<endl;

    

    H = findFundamentalMat(pt1,pt2,FM_RANSAC, 3,0.5);
    
//    cout << "Row:\t" << H.rows << endl;
//    cout << "Col:\t" << H.cols << endl;
//    cout << H << endl;
    
    // Get the corners from the image_1 ( the object to be "detected" )
    vector<Point2f> corners1(4);
    corners1[0] = cvPoint(0, 0);
    corners1[1] = cvPoint(img1.cols, 0);
    corners1[2] = cvPoint(img1.cols, img1.rows);
    corners1[3] = cvPoint(0, img1.rows);
    vector<Point2f> corners2(4);
    
    perspectiveTransform(corners1, corners2, H);
    
    // Draw lines between the corners (the mapped object in the scene - image_2 )
//    line(imgMatches, corners2[0] + Point2f(img1.cols, 0), corners2[1] + Point2f(img1.cols, 0), Scalar(255, 0, 0), 2);
//    line(imgMatches, corners2[1] + Point2f(img1.cols, 0), corners2[2] + Point2f(img1.cols, 0), Scalar(255, 0, 0), 2);
//    line(imgMatches, corners2[2] + Point2f(img1.cols, 0), corners2[3] + Point2f(img1.cols, 0), Scalar(255, 0, 0), 2);
//    line(imgMatches, corners2[3] + Point2f(img1.cols, 0), corners2[0] + Point2f(img1.cols, 0), Scalar(255, 0, 0), 2);
//    
    // Show detected matches
//    imshow("Good Matches & Object detection", imgMatches);
    
//    waitKey(0);
    return 0;
    
}


