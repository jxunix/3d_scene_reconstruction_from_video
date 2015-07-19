# 3d_scene_reconstruction_from_video
Advisor: Prof. Margrit Betke

1. Extracted SIFT features from each frame, took inlier set labelled by RANSAC and performed feature matching through FLANN. Coded in C++ using OpenCV;
2. Devised an algorithm that improved the tracking of the features in a long sequences of frames;
3. Computed the rotation and translation of the camera between each two frames via Epipolar Constraint and creatively resolved the solution ambiguities caused by sign;
4. Refined the estimation of feature depth iteratively by using small-baseline images and visualized the 3D features in Matlab, which were aligned with the real scene.
