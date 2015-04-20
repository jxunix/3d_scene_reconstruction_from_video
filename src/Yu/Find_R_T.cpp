//
//  Find_R_T.cpp
//  CS585_final_project
//
//  Created by Chen Yu on 11/9/14.
//  Copyright (c) 2014 Chen Yu. All rights reserved.
//

#include "Find_R_T.h"


int Find_R_T(Mat F, Mat_<double>& E, Mat& R, Mat_<double>& T){
    //-- Step 5: calculate Essential Matrix
    
    double Cam_Mat[] = {679.1762904713844, 0, 329.4494923815807,
        0, 681.184065752318, 201.8327112232685,
        0, 0,1};
    
    Mat K(3, 3, CV_64F, Cam_Mat);

    
    E = K.t() * F * K; //according to HZ (9.12)
    
    //-- Step 6: calculate Rotation Matrix and Translation Vector
    Mat P;
    Mat P1;
    //decompose E to P' , HZ (9.19)
    SVD svd(E,SVD::MODIFY_A);
    Mat svd_u = svd.u;
    Mat svd_vt = svd.vt;
    
    double sigma_data[] = {svd.w.at<double>(0), 0, 0,
                            0,svd.w.at<double>(1), 0,
                            0,0,svd.w.at<double>(2)};
    
    Mat svd_w = Mat(3, 3, CV_64F,sigma_data);
    
    
    
    double W_data[] = {0,-1,0,1,0,0,0,0,1};
    Mat W(3, 3, CV_64F,W_data);//HZ 9.13
    R = svd_u * Mat(W) * svd_vt; //HZ 9.19
    
    T = svd_u.col(2); //u3
    
    
//    T = svd_u * W * svd_w * svd_u.t();
//    cout << "test:" << svd_w <<endl;
    //    if (!CheckCoherentRotation (R)) {
    //        std::cout<<"resulting rotation is not coherent\n";
    //        P1 = 0;
    //        return 0;
    //    }
    
//    double P1_data[] = {R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),T(0),
//                        R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),T(1),
//                        R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2),T(2)};
//    
//    P1 = Mat(3, 3, CV_64F,P1_data);
//    
//    //-- Step 7: Reprojection Matrix and rectification data
//    Mat R1, R2, P1_, P2_, Q;
//    Rect validRoi[2];
//    double dist[] = { -0.03432, 0.05332, -0.00347, 0.00106, 0.00000};
//    Mat D(1, 5, CV_64F, dist);
    
//    stereoRectify(K, D, K, D, img_1.size(), R, t, R1, R2, P1_, P2_, Q, CV_CALIB_ZERO_DISPARITY, 1,
//    img_1.size(),  &validRoi[0], &validRoi[1] );


    return 0;
}
