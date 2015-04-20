//
//  Find_R_T.h
//  CS585_final_project
//
//  Created by Chen Yu on 11/9/14.
//  Copyright (c) 2014 Chen Yu. All rights reserved.
//

#ifndef __CS585_final_project__Find_R_T__
#define __CS585_final_project__Find_R_T__

#include <iostream>
#include <vector>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace std;
using namespace cv;


int Find_R_T(Mat F, Mat_<double>& E, Mat& R, Mat_<double>& T);


#endif /* defined(__CS585_final_project__Find_R_T__) */
