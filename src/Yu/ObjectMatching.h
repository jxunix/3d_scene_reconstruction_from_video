//
//  ObjectMatching.h
//  CS585_final_project
//
//  Created by Chen Yu on 11/8/14.
//  Copyright (c) 2014 Chen Yu. All rights reserved.
//

#ifndef CS585_final_project_ObjectMatching_h
#define CS585_final_project_ObjectMatching_h

#include <iostream>
#include <vector>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

#define SIFT
//#define SURF


int matching(string path1, string path2, Mat &H,
             vector<KeyPoint>& keypts1, vector<KeyPoint>& keypts2,
             Mat& descrs1, Mat& descrs2,
             vector<DMatch>& matches);

#endif
