//
//  TrackObject.h
//  CS585_final_project
//
//  Created by Chen Yu on 11/21/14.
//  Copyright (c) 2014 Chen Yu. All rights reserved.
//

#ifndef __CS585_final_project__TrackObject__
#define __CS585_final_project__TrackObject__

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <iomanip>  

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace std;
using namespace cv;

#define min_track_length 5

static vector< vector< Point2f > > good_point_tractor;

static vector< vector< Point2f > > tracking_path;

static Point2f bad_point = Point2f(-1,-1);

void TrackObject(string path1, string path2,
                 vector<KeyPoint>& keypts1, vector<KeyPoint>& keypts2,
                 Mat& descrs1, Mat& descrs2,
                 vector<DMatch>& matches);

void check_previous_point_matching( vector<DMatch>& matches,
                                   vector<KeyPoint>& keypts1,
                                   vector<KeyPoint>& keypts2,
                                   vector<Point2f> pt1,
                                   vector<Point2f> pt2);

void check_redundency();

void print_tracking_path();

void pick_good_tracking(int start_frame_index, int end_frame_index);

void initial_good_point_tractor();

void print_tracking_matrix();

void draw_matching_images(vector<int> img_index);

#endif /* defined(__CS585_final_project__TrackObject__) */
