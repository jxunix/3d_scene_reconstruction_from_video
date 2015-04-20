//  TrackObject.cpp
//  CS585_final_project
//
//  Created by Chen Yu on 11/21/14.
//  Copyright (c) 2014 Chen Yu. All rights reserved.

#include "TrackObject.h"


void TrackObject(string path1, string path2,
                 vector<KeyPoint>& keypts1, vector<KeyPoint>& keypts2,
                 Mat& descrs1, Mat& descrs2,
                 vector<DMatch>& matches){
    //////////////////////////////////////////////////////////////////////////////////////////
    
    Mat img1 = imread(path1, CV_LOAD_IMAGE_UNCHANGED);
    Mat img2 = imread(path2, CV_LOAD_IMAGE_UNCHANGED);
    
    putText(img1, path1.substr(path1.length()-8,path1.length()-6), Point(10,10), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255), 2.0);
    putText(img2, path2.substr(path2.length()-8,path2.length()-6), Point(10,10), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255), 2.0);
    
    FlannBasedMatcher matcher;
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
        if (matches[i].distance < 12 * minDist)
        {
            goodMatches.push_back(matches[i]);
        }
    }
    
    Mat imgMatches;
    
    // Localize the object
    vector<Point2f> pt1, pt2;
    for (int i = 0; i < goodMatches.size(); i++)
    {
        // Get the keypoints from the good matches
        pt1.push_back(keypts1[goodMatches[i].queryIdx].pt);
        pt2.push_back(keypts2[goodMatches[i].trainIdx].pt);
        
        // find the label of matching point
        ostringstream convert;
        convert << i;
        
        //        putText(img1, convert.str().c_str(), keypts1[goodMatches[i].queryIdx].pt, FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        circle(img1, keypts1[goodMatches[i].queryIdx].pt,2, Scalar(0,255,0),CV_FILLED, 8,0);
        
        //        putText(img2, convert.str().c_str(), keypts2[goodMatches[i].trainIdx].pt, FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        circle(img2, keypts2[goodMatches[i].trainIdx].pt,2, Scalar(0,255,0),CV_FILLED, 8,0);
    }
    
    
    //////////////////////////////////////////////////////////////////////////////////////////
    
    check_previous_point_matching(matches, keypts1, keypts2,pt1, pt2 );
    
    check_redundency();
    
    //    print_tracking_matrix();
    
    //////////////////////////////////////////////////////////////////////////////////////////
    
        Mat imgMatches2;
        drawMatches(img1, keypts1, img2, keypts2, goodMatches, imgMatches2, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
        imshow("label img1", img1);
        imshow("label img2", img2);
        imshow("Good Matches & Object detection", imgMatches2);
        imwrite("/Users/chenyu/Dropbox/CS585_Project/Albert Code/Dec-04/neighbour_matching2.jpg", imgMatches2);
    
//    vector<int> img_index;
//    img_index.push_back(1);
//    img_index.push_back(2);
//    img_index.push_back(3);
////
//    draw_matching_images(img_index);
    
    //////////////////////////////////////////////////////////////////////////////////////////
    
//    waitKey(0);
}

void pick_good_tracking(int start_frame_index, int end_frame_index){
    for (int ii = 0; ii < good_point_tractor.size(); ii++) {
        bool good_track = true;
        for (int jj = start_frame_index ; jj <= end_frame_index+ 1; jj++) {
            if (good_point_tractor[ii][jj] == bad_point) {
                good_track = false;
            }
        }
        if (good_track) {
            tracking_path.push_back(good_point_tractor[ii]);
        }
    }
    
    print_tracking_path();
}


void draw_matching_images(vector<int> img_index){
    if (img_index.size() > good_point_tractor.size()) {
        cout<<"need more tracking to show sequence images\n";
        return;
    }
    
    vector<Mat> img_buff;
    //
    for (int ii = 0; ii < img_index.size(); ii++) {
        ostringstream convert;
        
//        convert << "/Users/chenyu/Dropbox/CS585_Project/Tests/2014-12-02_Office_Scene/image_" <<
//        setfill('0') << setw(4) << img_index[ii] << ".jpg";
        
        convert << "/Users/chenyu/Dropbox/CS585_Project/Tests/2014-11-04_Desk_Scene/image_" <<
        setfill('0') << setw(4) << img_index[ii] << ".jpg";
        
        String path = convert.str().c_str();
        Mat img_tmp = imread(path, CV_LOAD_IMAGE_UNCHANGED);
        img_buff.push_back(img_tmp);
        
        putText(img_tmp, path.substr(path.length()-8,path.length()-6), Point(10,10), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255), 2.0);
        convert<<"";
    }
    
    Mat combine = Mat::zeros(img_buff[0].rows, img_buff[0].cols * (int)img_index.size(), img_buff[0].type());
    int cols = img_buff[0].cols;
    for (int i=0;i<combine.cols;i++) {
        int fram_index = i / img_buff[0].cols;
        cout<<fram_index<<endl;
        img_buff[fram_index].col(i % cols).copyTo(combine.col(i));
    }
    
    pick_good_tracking(img_index[0], img_index.back());
    
    
    
    for (int kk = 0; kk < tracking_path.size(); kk++) {
        for (int pp = 0; pp < img_index.size() -1 ; pp++) {
            line( combine,
                 
                 Point(tracking_path[kk][img_index[pp]].y + pp *  cols,
                       tracking_path[kk][img_index[pp]].x),
                 
                 Point(tracking_path[kk][img_index[pp+1]].y + (pp+1) * cols,
                       tracking_path[kk][img_index[pp+1]].x),
                 
                 Scalar( kk * 111 % 255, kk * 222 % 255, kk * 55 % 255 ),  1, 8 );
            
//        circle( combine,
//               Point(tracking_path[kk][img_index[pp]].x,
//                     tracking_path[kk][img_index[pp]].y + pp *  cols),
//               1, Scalar( 0, 0, 255 ), 1, 8 );
        }
    }
    
    for (int kk = 0; kk < tracking_path.size(); kk++) {
        for (int pp = 0; pp < img_index.size(); pp++) {
            
                    circle( combine,
                           Point(tracking_path[kk][img_index[pp]].y + pp *  cols,
                                 tracking_path[kk][img_index[pp]].x),
                           2, Scalar( 0, 0, 255 ), 1, 8 );
        }
    }
    
    Size size(combine.cols/2, combine.rows/2);//the dst image size,e.g.100x100
    resize(combine,combine,size);//resize image
    
    imshow("matching plot", combine);
    imwrite("/Users/chenyu/Dropbox/CS585_Project/Albert Code/Dec-04/tracking_path.jpg", combine);
     waitKey(0);
}

void check_previous_point_matching( vector<DMatch>& matches,
                                   vector<KeyPoint>& keypts1,
                                   vector<KeyPoint>& keypts2,
                                   vector<Point2f> pt1,
                                   vector<Point2f> pt2){
    
    printf("\n\ntracking object: keypoint:%lu  goodpoint:%lu\n",keypts1.size(), pt1.size());
    
    if (good_point_tractor.size() == 0) {
        for (int ii = 0; ii < pt1.size(); ii++) {
            vector< Point2f > header;
            header.push_back(pt1.at(ii));
            header.push_back(pt2.at(ii));
            good_point_tractor.push_back(header);
        }
        return;
    }
    
    unsigned long frame_index = good_point_tractor[0].size();
    unsigned long good_point_tractor_size = good_point_tractor.size();
    
    for (int jj= 0; jj < pt1.size(); jj++) {
        Point2f source = pt1[jj];
        
        bool add_new_line = true;
        for (int ii = 0; ii < good_point_tractor_size; ii++) {
            
            if (source == good_point_tractor[ii].back()) {
                good_point_tractor[ii].push_back(pt2[jj]);
                //                cout << "find a matching point:"<< ii;
                //                printf("(%d,%d) \n", (int)source.x, (int)source.y );
                add_new_line = false;
                break;
            }
        }
        
        if (add_new_line) {
            vector<Point2f> newline;
            //            cout << "add new object\n";
            for (int k = 0; k < frame_index; k++) {
                newline.push_back(bad_point);
                //                cout << 2 <<endl;
            }
            newline.push_back(pt2[jj]);
            good_point_tractor.push_back(newline);
        }
    }
    
    for (int ii = 0; ii < good_point_tractor_size ; ii++) {
        if (good_point_tractor[ii].size() == frame_index) {
            good_point_tractor[ii].push_back(bad_point);
            //            cout <<"make up matirx:" <<ii<<endl;
        }
    }
    
}



void check_redundency(){
    for (int ii = 0; ii < good_point_tractor.size(); ii++) {
        int good_obj_counter = 0;
        
        for (int jj = 0; jj < good_point_tractor[ii].size(); jj++) {
            if (good_point_tractor[ii][jj] != bad_point) {
                good_obj_counter++;
            }
            //            cout<< "object counting:"<<good_obj_counter<<endl;
        }
        
        if (good_obj_counter <= min_track_length && good_point_tractor[ii].back() == bad_point) {
            good_point_tractor.erase(good_point_tractor.begin() + ii);
            //            cout<< "delete object tracking"<<endl;
            ii--;
        }
    }
}

void print_tracking_matrix(){
    if (good_point_tractor.size() == 0) {
        cout<< "empty good_point_tractor\n";
        return;
    }
    
    for (int ii = 0; ii < good_point_tractor.size(); ii++) {
        printf("%3d:",ii);
        for (int jj = 0; jj < good_point_tractor.at(0).size(); jj++) {
            Point2f tem_point = good_point_tractor.at(ii).at(jj);
            printf("(%d,%d) ",  (int)tem_point.x, (int)tem_point.y );
        }
        cout<<""<<endl;
    }
}

void print_tracking_path(){
    if (tracking_path.size() == 0) {
        cout<< "empty tracking_path\n";
        return;
    }
    
    for (int ii = 0; ii < tracking_path.size(); ii++) {
        printf("%3d:",ii);
        for (int jj = 0; jj < tracking_path.at(0).size(); jj++) {
            Point2f tem_point = tracking_path.at(ii).at(jj);
            printf("(%d,%d) ",  (int)tem_point.x, (int)tem_point.y );
        }
        cout<<""<<endl;
    }
}
