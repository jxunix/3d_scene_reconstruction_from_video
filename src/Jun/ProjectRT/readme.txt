We use Microsoft Visual Studio. The solution and project are both called ProjectRT.

To compile the program, make sure the project properties are correct, including "C/C++ - General - Additional Include Directories", "Linker - General - Addiional Library Directories", and "Linker - Input - Additional Dependencies". They are all related the OpenCV library.

To run the program, please copy folder "2014-12-02_Office_Scene" to directory "ProjectRT/ProjectRT".

Some parameters worth noticing:
(1) FRAME_NUM, 30 is the number of frames we will process in the program.
(2) BASIC_INTERVAL = 4 means for every 4 frames we compute the relative displacement because in that case the variance of the estimation is small enough.

Program design:
(1) Read image i and image (i+4);
(2) Extract the SIFT features and also their descriptions;
(3) Match two sets of SIFT features by FLANN based matching;
(4) Remove them if the distance between matched features is larger than 3 times the minimum;
(5) Find the fundamental matrix through RANSIC and exclude outliers obtained from RANSAC;
(6) Draw combined two frames (optional);
(7) Compute all possible rotation matrices and translation vectors from the essential matrix via SVD;
(8) Determine the true value of rotation matrix, refer to the report for more infomation;
(9) Compute lambdas, the scale factor by SVD;
(10) Estimate the scene coordinates in the left camera coordinate system.