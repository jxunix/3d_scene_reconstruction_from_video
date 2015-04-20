#include <iostream>
#include <vector>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "ObjectMatching.h"
#include "Find_R_T.h"
#include "TrackObject.h"

using namespace std;
using namespace cv;

int matching(string path1, string path2);

int main(int argc, char** argv)
{
//////////////////////////////////////////////////////////////////////////////////////////

    int begin_index = 1;
    int end_index = 20;
    
    for (int ii = begin_index; ii <= end_index ; ii++) {
        ostringstream convert1;
//        convert1 << "/Users/chenyu/Dropbox/CS585_Project/Tests/2014-12-02_Office_Scene/image_" << setfill('0') << setw(4) << ii << ".jpg";
        convert1 << "/Users/chenyu/Dropbox/CS585_Project/Tests/2014-11-04_Desk_Scene/image_" << setfill('0') << setw(4) << ii << ".jpg";
        
        String path1 = convert1.str().c_str();
        ostringstream convert2;
//        convert2 << "/Users/chenyu/Dropbox/CS585_Project/Tests/2014-12-02_Office_Scene/image_"<< setfill('0') << setw(4) << ii+1 << ".jpg";
        convert2 << "/Users/chenyu/Dropbox/CS585_Project/Tests/2014-11-04_Desk_Scene/image_"<< setfill('0') << setw(4) << ii+1 << ".jpg";
        String path2 = convert2.str().c_str();
        
        
        Mat F;
        Mat H;
        vector<KeyPoint> keypts1;
        vector<KeyPoint> keypts2;
        Mat descrs1;
        Mat descrs2;
        vector<DMatch> matches;

        matching( path1, path2, F, keypts1, keypts2, descrs1, descrs2, matches);
        Mat_<double> E,R,T;
        Find_R_T(F, E, R, T);
        
//////////////////////////////////////////////////////////////////////////////////////////
        
        TrackObject(path1, path2, keypts1, keypts2, descrs1, descrs2, matches);
        
    }
    
    vector<int> img_index;
    img_index.push_back(1);
//    img_index.push_back(2);
    img_index.push_back(3);
//    img_index.push_back(4);
    img_index.push_back(5);
//    img_index.push_back(6);
    img_index.push_back(10);
    
    draw_matching_images(img_index);
//    print_tracking_matrix();
    
    //Difference of Gaussian is obtained as the difference of Gaussian blurring of an image with two different \sigma
    
    return 0;
}



