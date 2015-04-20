%--------------------------------------------------------------------------
%
% File Name:      reconstruction.m
% Date Created:   2014/11/09
% Date Modified:  2014/12/07
%
% Author:         Eric Cristofalo
% Contact:        eric.cristofalo@gmail.com
%
% Description:    3D Reconstruction From Sequence of Images
%                 
%                 This script represented the entire 3D reconstruction
%                 algorithm that builds a primative 3D model from a set of
%                 images defined in the begining. There are several
%                 functions used to make this code that may cause the
%                 script to fail if they don't exist, however they are
%                 available upon request. If the reconstruction does not
%                 work or does not make sense, first check which original
%                 estimate of R and T are being used (1 or 2). Otherwise,
%                 the scene may be too general (not enough depth) in order
%                 for a proper reconstruction. 
%
% Inputs:         Requires a camera calibration matrix (K) and camera
%                 distortion coefficients. Also requires input images in
%                 some directory.
%
% Outputs:        Script diplays the original images and the matches
%                 features among them. The depth estimation is displayed
%                 over the original images as colored dots to represent a
%                 depth image similar to RGB-D cameras. Finally the 3D
%                 reconstructions are plotted in a 3D plot which diplays
%                 the 3D model and the locations/orientations of each
%                 camera from the image sequence. 
%
% Example:
%
%--------------------------------------------------------------------------

figure(1); clf(1);   % Feature Matches
figure(2); clf(2);   % 2D Depth Visualization
figure(10); clf(10); % 3D Reconstruction
clear all;
clc;

%% Initialization

% Our Vision Functions
addpath(genpath('/Users/ericcristofalo/Dropbox/BU/Shared/Homography_Formations/Vision_Code'));
addpath(genpath('/Users/ericcristofalo/Dropbox/BU/Research/2014_Active_Estimation/3D_Depth_Estimation'));

% Camera Calibration (TV-IP672W 640x400 Excellent Quality)
cam.cameraMatrix = [679.1762904713844 0                329.4494923815807;
                    0                 681.184065752318 201.8327112232685;
                    0                 0                1.0000           ];
cam.distCoeffs = [-0.2461283687431849    ;
                   0.2816828141575686    ;
                   0.0002154787123809983 ;
                  -0.001189677880083738  ;
                  -0.3734268366833506    ]';

% Camera Calibration (Eric's Macbook Pro)
% cam.cameraMatrix = [1045.312752321511 0                 628.7914280476415;
%                     0                 1041.764314953895 360.6971022774602;
%                     0	              0                 1];
% cam.distCoeffs = [-0.2461283687431849    ;
%                    0.2816828141575686    ;
%                    0.0002154787123809983 ;
%                   -0.001189677880083738  ;
%                   -0.3734268366833506    ]';

cam.K = cam.cameraMatrix;
cam.K_inv = inv(cam.K);
cam.imSize = [400, 640, 3];

% Image Path
cam.imPath = '/Users/ericcristofalo/Dropbox/BU/Shared/CS585_Project/Tests/';
% cam.imSet = '2014-11-04_Desk_Scene/';
cam.imSet = '2014-12-02_Office_Scene/';

% Video Sequence
cam.startFrame = 40; % first frame in sequence
cam.frameInt = 4;    % every xth frame
cam.totalImages = 4; % total images in sequence
cam.images = ...     % range of test images
   [cam.startFrame,cam.startFrame+(cam.totalImages-1)*cam.frameInt];
cam.imRange = cam.images(1):cam.frameInt:cam.images(end);

% Matches
vis.matcher = cv.DescriptorMatcher('BruteForce');
% vis.matcher = cv.DescriptorMatcher('FlannBased');
master.ind = ones(10000,length(cam.imRange))*(-1);
master.descriptors = ones(10000,128,length(cam.imRange))*(-1);
vis.imMatches = ...
   uint8(zeros(cam.imSize(1), cam.imSize(2)*cam.totalImages, 3));

% Post Processing
post.imPoints = ones(4,10000,length(cam.imRange))*(-1);
post.coords = zeros(6,cam.totalImages);
post.coordsPlot = zeros(6,cam.totalImages);
post.R = zeros(3,3,cam.totalImages);
post.R(:,:,1) = eye(3);
post.T = zeros(3,1,cam.totalImages);


%% Matching Features

for imInd = 1:length(cam.imRange)-1
   
   % Image Indices
   im1.i = cam.imRange(imInd);
   im2.i = cam.imRange(imInd+1);

   % Read Input Images
   im1.name = ['image_',num2str(sprintf('%04.0f',im1.i)),'.jpg'];
   im2.name = ['image_',num2str(sprintf('%04.0f',im2.i)),'.jpg'];
   
   % Keypoint and Descriptor Extraction
   % Previous Image
   if imInd==1
      im1.im = imread([cam.imPath,cam.imSet,im1.name]);
      im1.im = cv.undistort(im1.im, cam.K, cam.distCoeffs);
      [im1.keypoints,im1.descriptors] = cv.SIFT(im1.im);
   else % set 1st image to old image
      im1.im = im2.im;
      im1.keypoints = im2.keypoints;
      im1.descriptors = im2.descriptors;
   end
   % New Image
   im2.im = imread([cam.imPath,cam.imSet,im2.name]);
   im2.im = cv.undistort(im2.im, cam.K, cam.distCoeffs);
   [im2.keypoints,im2.descriptors] = cv.SIFT(im2.im);
   
   % [im1.keypoints,im1.descriptors] = cv.SIFT(im1.im,...
   %    'ConstrastThreshold',0.05,'EdgeThreshold',5.0);
   % [im2.keypoints,im2.descriptors] = cv.SIFT(im2.im,...
   %    'ConstrastThreshold',0.05,'EdgeThreshold',5.0);
   
   % Keypoint Matching
   vis.matches = vis.matcher.match(im1.descriptors,im2.descriptors);
   vis.matchesGood = zeros(length(vis.matches),4);
   for i = 1:length(vis.matches);
      vis.matchesGood(i,1) = vis.matches(i).queryIdx;
      vis.matchesGood(i,2) = vis.matches(i).trainIdx;
      vis.matchesGood(i,3) = vis.matches(i).imgIdx;
      vis.matchesGood(i,4) = vis.matches(i).distance;
   end
      
   % Refine Matches
   if imInd==1 % remove matches with poor error
      vis.matchesGood(vis.matchesGood(:,4)>=150,:) = [];
   else
      vis.matchesGood(vis.matchesGood(:,4)>=150,:) = [];
   end
   
   % Match Points
   vis.points1 = zeros(length(vis.matchesGood),2);
   vis.points2 = zeros(length(vis.matchesGood),2);
   for i = 1:length(vis.matchesGood);
      vis.points1(i,:) = im1.keypoints(vis.matchesGood(i,1)+1).pt;
      vis.points2(i,:) = im2.keypoints(vis.matchesGood(i,2)+1).pt;
   end
   vis.points1 = reshape(vis.points1,[1,length(vis.points1),2]);
   vis.points2 = reshape(vis.points2,[1,length(vis.points2),2]);
   
   % Fundamental Matrix
   [vis.F, vis.mask] = cv.findFundamentalMat(vis.points1,vis.points2,...
      'Method','Ransac','Param1',3,'Param2',0.999);
   if imInd==1 % save first Fundamental guess
      post.F = vis.F;
   end
   
   % Finding Inliers
   vis.maskInd = find(vis.mask~=0);
   vis.matchesInliers = zeros(length(vis.maskInd),4);
   for i = 1:length(vis.maskInd)
      ind = vis.maskInd(i);
      vis.matchesInliers(i,:) = vis.matchesGood(ind,:);
   end 
   
   % Save to Master
   extra = 10000-length(vis.matchesInliers);
   if imInd==1
      % Matches
      master.descriptors(:,:,1) = ...
         [im1.descriptors;(-1)*ones(10000-length(im1.descriptors),128)];
      master.ind(:,1) = [vis.matchesInliers(:,1);(-1)*ones(extra,1)];
      master.descriptors(:,:,2) = ...
         [im2.descriptors;(-1)*ones(10000-length(im2.descriptors),128)];
      master.ind(:,2) = [vis.matchesInliers(:,2);(-1)*ones(extra,1)];
      % Points
      for i = 1:length(vis.matchesInliers)
         curInd = vis.matchesInliers(i,1);
         position = im1.keypoints(curInd+1).pt;
         post.imPoints(:,i,1) = [position';1;curInd];
         curInd = vis.matchesInliers(i,2);
         position = im2.keypoints(curInd+1).pt;
         post.imPoints(:,i,2) = [position';1;curInd];
      end
   else
      % Descriptors
      master.descriptors(:,:,imInd+1) = ...
         [im2.descriptors;(-1)*ones(10000-length(im2.descriptors),128)];
      
      % Generating master.ind for Matches
      for i = 1:length(vis.matchesInliers)
         matched=0;
         for j = 1:length(master.ind)
            curInd = vis.matchesInliers(i,1);
            prevInd = master.ind(j,imInd);
            if curInd==prevInd && matched==0;
               master.ind(j,imInd+1) = vis.matchesInliers(i,2);
               matched = 1;
            end
         end
         if matched==0
            newInd = min(find(sum(master.ind,2)==-size(master.ind,2)));
            master.ind(newInd,imInd:imInd+1) = vis.matchesInliers(i,1:2);
         end
      end
      
      % Points
      for i = 1:length(vis.matchesInliers)
         curInd = vis.matchesInliers(i,2);
         position = im2.keypoints(curInd+1).pt;
         post.imPoints(:,i,imInd+1) = [position';1;curInd];
      end
      
   end
   completeMatches = sum(ismember(master.ind(:,1:imInd+1),-1),2)==0;
   master.numMatches = sum(completeMatches);
   master.matches = master.ind(completeMatches,:);
   
   % Output Matches
   masterDisplay = master.ind;
   masterDisplay(sum(master.ind,2)==-size(master.ind,2),:)=[];
   
   % Save Matched Image
   if imInd==1
      % Save Matches for Plotting
      post.im1.im = im1.im;
      post.im2.im = im2.im;
      post.im1.keypoints = im1.keypoints;
      post.im2.keypoints = im2.keypoints;
      post.matchesInliers = vis.matchesInliers;
      % Create Image
      for i = 1:3;
         vis.imMatches(:,(imInd-1)*cam.imSize(2)+1:(imInd)*cam.imSize(2),i)...
            = im1.im(:,:,i);
      end
   end
   for i = 1:3;
      vis.imMatches(:,(imInd)*cam.imSize(2)+1:(imInd+1)*cam.imSize(2),i)...
         = im2.im(:,:,i);
   end

end

% Save Image Points for 8-Point Estimation
actualPoints = sum(ismember(master.ind(:,1:2),-1),2)==0; % remove -1 placeholders
actualMatches = master.ind(actualPoints,:,1);
post.imPointsMatches_12 = zeros(4,length(actualMatches),2);
for i = 1:length(actualMatches)
   % Find Match Index
   matchInd1 = actualMatches(i,1);
   matchInd2 = actualMatches(i,2);
   % Pull Out Matching Feature Pairs
   post.imPointsMatches_12(:,i,1) = ...
      post.imPoints(:,post.imPoints(4,:,1)==matchInd1,1);
   post.imPointTemp = post.imPoints(:,post.imPoints(4,:,2)==matchInd2,2);
   post.imPointsMatches_12(:,i,2) = post.imPointTemp(:,1); % incase of duplicates
end

% Save Image Points for Depth Refinement
post.imPointsMatches = zeros(4,master.numMatches,cam.totalImages);
for k = 1:cam.totalImages
   for i = 1:master.numMatches
      % Find Match Index
      matchInd = master.matches(i,k);
      % Pull Out Matching Feature
      post.imPointTemp = post.imPoints(:,post.imPoints(4,:,k)==matchInd,k);
      post.imPointsMatches(:,i,k) = post.imPointTemp(:,1); % incase of duplicates
   end
end


%% Initial 8-Point Epipolar Depth Estimation (on first Iteration only)

% Essential Matrix
est.E = cam.K'*post.F*cam.K;
[est.U_E, est.S_E, est.V_E] = svd(est.E);
est.Sig = diag([1;1;0]);
% Sig = (S_E(1,1)+S_E(2,2))/2*diag([1;1;0]); % from textbook proof
est.E_proj = est.U_E*est.Sig*est.V_E';

% Relative Rotation and Translation
est.R1 = est.U_E*rotMat('z',pi/2)'*est.V_E';
est.T1_x = est.U_E*rotMat('z',pi/2)*est.Sig*est.U_E';
est.T1 = [-est.T1_x(2,3); est.T1_x(1,3); -est.T1_x(1,2)];
est.R2 = est.U_E*rotMat('z',-pi/2)'*est.V_E';
est.T2_x = est.U_E*rotMat('z',-pi/2)*est.Sig*est.U_E';
est.T2 = [-est.T2_x(2,3); est.T2_x(1,3); -est.T2_x(1,2)];

% Least Squares Depth Estimation
est.numMatches = size(post.imPointsMatches_12,2);
est.M1 = zeros(3*est.numMatches, est.numMatches+1);
est.M2 = est.M1;
for i = 1:est.numMatches
   ind = i*3-2;
   % Camera Frame Points
   x1 = cam.K_inv*post.imPointsMatches_12(1:3,i,1);
   x2_x = skewSymMat(cam.K_inv*post.imPointsMatches_12(1:3,i,2));
   % M Matrix
   est.M1(ind:ind+2,i) = x2_x*est.R1*x1;
   est.M1(ind:ind+2,end) = x2_x*est.T1;
   est.M2(ind:ind+2,i) = x2_x*est.R2*x1;
   est.M2(ind:ind+2,end) = x2_x*est.T2;
end

% SVD of M
[est.U_M1, est.S_M1, est.V_M1] = svd(est.M1);
est.lambda_1 = est.V_M1(:,end);
[est.U_M2, est.S_M2, est.V_M2] = svd(est.M2);
est.lambda_2 = est.V_M2(:,end);

% Compute Camera Coordiantes
est.T_gamma1 = est.T1*est.lambda_1(end);
est.T_gamma2 = est.T2*est.lambda_2(end);
% T_gamma = T*-1; % also a unit vector
for i = 1:est.numMatches
   x_1 = post.imPointsMatches_12(1:3,i,1);
   im1.camPoints1(:,i) = ...
      est.lambda_1(i)*cam.K_inv*x_1;
   im2.camPoints1(:,i) = ...
      est.lambda_1(i)*est.R1*cam.K_inv*x_1+est.T_gamma1;
   im1.camPoints2(:,i) = ...
      est.lambda_2(i)*cam.K_inv*x_1;
   im2.camPoints2(:,i) = ...
      est.lambda_2(i)*est.R2*cam.K_inv*x_1+est.T_gamma2;
end

est.summary = [(1:est.numMatches)',...
   im1.camPoints1(3,:)',im2.camPoints1(3,:)',...
   im1.camPoints2(3,:)',im2.camPoints2(3,:)'];
est.summary

% if sum(depthSummary(:,2))>sum(depthSummary(:,4))
%    im1.camPoints = im1.camPoints1;
%    im2.camPoints = im2.camPoints1;
%    disp('Chose R1 and T1');
% else
%    im1.camPoints = im1.camPoints2;
%    im2.camPoints = im2.camPoints2;
%    disp('Chose R2 and T2');
% end

im1.camPoints = im1.camPoints2;
im2.camPoints = im2.camPoints2;
disp('Chose R2 and T2');
post.R(:,:,2) = est.R2;
post.T(:,:,2) = est.T_gamma2;


%% Depth Estimation Refinement with Multiple Images

% Find Lambdas Corresponding to Final Matches Only
est.lambda = -est.lambda_2;
post.lambda = zeros(master.numMatches+1,1);
for i = 1:master.numMatches
   curInd = master.matches(i,1);
   lamInd = find(post.imPointsMatches_12(4,:,1)==curInd);
   post.lambda(i,1) = est.lambda(lamInd,1);
end
post.lambda(end,1) = est.lambda(end);

% While error is large and there are at least 3 images... 
err.reprojDiff = 1E6; % meters
err.imPointsMatches = post.imPointsMatches;
err.ind = 1;
err.alpha_final = post.lambda.^(-1);
err.lambda = post.lambda;
while err.reprojDiff > (0.01) && cam.totalImages>=3 && err.ind<100
   
   % Initialize New Lambdas
   err.alpha_final = [err.alpha_final,zeros(size(post.lambda,1),1)];
   err.alpha_cur = zeros(size(post.lambda));
   err.lambda = [err.lambda,zeros(size(post.lambda,1),1)];
   
   for k = 2:cam.totalImages

      % Assemble P_k Matrix
      P_k = zeros(3*master.numMatches,12);
      for i = 1:master.numMatches
         % Pull Out Matching Feature Pairs
         x_1Cur = post.imPointsMatches(1:3,i,1);
         x_kCur = post.imPointsMatches(1:3,i,k);
         % Compute Camera Coordiantes (without depth scale)
         x_1Cur = cam.K_inv*x_1Cur;
         x_kCur = cam.K_inv*x_kCur;
         % Assemble P_k Matrix
         x_kCur_x = skewSymMat(x_kCur);
         x_k1_kron = kron(x_1Cur',x_kCur_x);
         P_k(i*3-2:i*3,:) = [x_k1_kron, ...
            err.alpha_final(i,err.ind)*x_kCur_x];
      end
      
      % Least Squares Estimate of R_k1 and T_k1
      [err.U_P, err.S_P, err.V_P] = svd(P_k);
      err.R_est = [err.V_P(1,end),err.V_P(4,end),err.V_P(7,end);
                   err.V_P(2,end),err.V_P(5,end),err.V_P(8,end);
                   err.V_P(3,end),err.V_P(6,end),err.V_P(9,end)];
      err.T_est = err.V_P(10:12,end);
      [err.U,err.S,err.V] = svd(err.R_est);
      post.R(:,:,k) = sign(det(err.U*err.V'))*err.U*err.V';
      post.T(:,:,k) = sign(det(err.U*err.V'))*err.T_est/(det(err.S))^(1/3);
%       post.R
%       post.T
      
      % Compute New Scale Factors
      for i = 1:master.numMatches
         % Pull Out Matching Feature Pairs
         x_1Cur = post.imPointsMatches(1:3,i,1);
         x_kCur = post.imPointsMatches(1:3,i,k);
         % Compute Camera Coordiantes (without depth scale)
         x_1Cur = cam.K_inv*x_1Cur;
         x_kCur = cam.K_inv*x_kCur;
         % Compute New lambda Estimates
         x_kCur_x = skewSymMat(x_kCur);
         err.alpha_cur(i) = ...
            (x_kCur_x*post.T(:,:,k))'*x_kCur_x*post.R(:,:,k)*x_1Cur;
         err.alpha_denom = norm(x_kCur_x*post.T(:,:,k))^2;
         err.alpha_cur(i) = err.alpha_cur(i)/err.alpha_denom;
      end
            
      % Sum New Alpha Scale Estimates
      err.alpha_final(:,err.ind+1) = err.alpha_final(:,err.ind)+err.alpha_cur;
      
   end
   
   % Scales
   err.alpha_final(:,err.ind+1) = -err.alpha_final(:,err.ind+1);
   % Normalize
   err.alpha_final(:,err.ind+1) = err.alpha_final(:,err.ind+1)./err.alpha_final(1,err.ind+1);
   err.lambda(:,err.ind+1) = err.alpha_final(:,err.ind+1).^(-1);
   
   % Reprojection Error
   err.reprojDiff = 0;
   for k = 2:cam.totalImages
      for i = 1:master.numMatches
         x_kCur = cam.K_inv*post.imPointsMatches(1:3,i,k);
         X_1Cur = err.lambda(i,err.ind+1)*cam.K_inv*post.imPointsMatches(1:3,i,1);
         % Reproject Points
         err.imPointsMatches(1:3,i,k) = ...
            post.R(:,:,k)*X_1Cur+post.T(:,:,k);
         % Reprojection Error Comparison
         err.reprojDiff = err.reprojDiff + ...
            1/(cam.totalImages*master.numMatches)*...
            norm(x_kCur-[err.imPointsMatches(1:2,i,k);1])^2;
      end
   end
   
   disp(['Reprojection Error: ',num2str(err.reprojDiff)]);
   
   % Loop Counter
   err.ind = err.ind+1;
   
end


%% 3D Reconstruction Visualization

% Generate Normalized Depth For Colors on Plots
im1.maxDepth = max(abs(im1.camPoints(3,:)));
im2.maxDepth = max(abs(im2.camPoints(3,:)));
im1.normDepth = abs(im1.camPoints(3,:))./im1.maxDepth;
im2.normDepth = abs(im2.camPoints(3,:))./im2.maxDepth;

% Global Rotation of All Plotting Data
R_plot = rotMat('x',-pi/2);

% World w.r.t. Camera 1 with Final Lambda Estimates
figure(10); clf(10);
hold on;
% Coordinate Systems
post.labelOffset = 0.001;
for i = 1:cam.totalImages
   if i~=1
      post.coords(:,i) = [R_plot*post.T(:,:,i);rot2euler(-post.R(:,:,i))];
   end
   post.coordsPlot(:,i) = ...
      [post.coords(1:3,i);rot2euler(R_plot*post.R(:,:,i))];
   plotCoordSys(post.coordsPlot(:,i), 1 , [0,0,0], 0.01, 2.5)
   text(post.coordsPlot(1,i)+post.labelOffset,...
      post.coordsPlot(2,i)+post.labelOffset,...
      post.coordsPlot(3,i)+post.labelOffset,...
      num2str(i),'Color','r','EdgeColor','r','LineWidth',1.5,'Margin',0.5);
end
% 3D Reconstructions
for i = 1:master.numMatches % new lambda values
   x_1 = post.imPointsMatches(1:3,i,1);
   post.camPoints(:,i) = ...
      est.lambda(i,end)*cam.K_inv*x_1;
end

if sum(post.camPoints(3,:))<0 % flip the signs
   post.camPoints = -post.camPoints;
end
% Rotate Points into Plot Perspective
post.camPointsPlot(1:3,:) = R_plot*post.camPoints(1:3,:);
for i = 1:master.numMatches
   d = im1.normDepth(i);
   scatter3(post.camPointsPlot(1,i),post.camPointsPlot(2,i),post.camPointsPlot(3,i),...
      60,[d/2,d,(1-d)],'Fill');
%    text(post.camPointsPlot(1,i),post.camPointsPlot(2,i),post.camPointsPlot(3,i),...
%       num2str(post.imPointsMatches(4,i)),'Color',[1,0.5,0]);
end
title(['3D Reconstruction: ',num2str(cam.totalImages),' Images and 100 Iterations']);
xlabel('x-axis (m)'); ylabel('y-axis (m)'); zlabel('z-axis (m)');
axis equal; %axis([-0.1 0.1 -0.1 0.1 0 0.2]);
% box on;
hold off;


% World w.r.t. Camera 1 From First 2 Images ONLY
figure(11); clf(11);
hold on;
% Coordinate Systems
post.labelOffset = 0.001;
for i = 1:2
   if i~=1
      post.coords(:,i) = [-post.T(:,:,i);rot2euler(-post.R(:,:,i))];
   end
   post.coordsPlot(:,i) = ...
      [post.coords(1:3,i);rot2euler(R_plot*post.R(:,:,i))];
   plotCoordSys(post.coordsPlot(:,i), 1 , [0,0,0], 0.01, 2.5)
   text(post.coordsPlot(1,i)+post.labelOffset,...
      post.coordsPlot(2,i)+post.labelOffset,...
      post.coordsPlot(3,i)+post.labelOffset,...
      num2str(i),'Color','r','EdgeColor','r','LineWidth',1.5,'Margin',0.5);
end
% 3D Reconstructions
if sum(im1.camPoints(3,:))<0 % flip the signs
   im1.camPoints = -im1.camPoints;
end
% Rotate Points into Plot Perspective
im1.camPointsPlot(1:3,:) = R_plot*im1.camPoints(1:3,:);
for i = 1:size(post.imPointsMatches_12,2)
   d = im1.normDepth(i);
   scatter3(im1.camPointsPlot(1,i),im1.camPointsPlot(2,i),im1.camPointsPlot(3,i),...
      60,[d/2,d,(1-d)],'Fill');
%    text(im1.camPointsPlot(1,i),im1.camPointsPlot(2,i),im1.camPointsPlot(3,i),...
%       num2str(post.imPointsMatches_12(4,i)),'Color',[1,0.5,0]);
end
title('3D Reconstruction: 2 Images and 0 Iterations');
xlabel('x-axis (m)'); ylabel('y-axis (m)'); zlabel('z-axis (m)');
axis equal; %axis([-0.1 0.1 -0.1 0.1 0 0.2]);
% box on;
hold off;


%% Plot Matches

% Drawing Final Matches
figure(1); imshow(vis.imMatches);
title('Total Image Matches');
hold on;
for i = 1:master.numMatches
   randColor = rand(1,3);
   curInd = zeros(cam.totalImages,1);
   curPoint = zeros(cam.totalImages,2);
   for k = 1:cam.totalImages
      curInd(k) = master.matches(i,k);
      point = post.imPoints(1:2,post.imPoints(4,:,k)==curInd(k),k);
      curPoint(k,:) = point(:,1)';
      scatter(curPoint(k,1)+(k-1)*cam.imSize(2), curPoint(k,2),...
         50, randColor, 'Fill');
      curPoint(k,1) = curPoint(k,1)+(k-1)*cam.imSize(2);
   end
   line(curPoint(:,1),curPoint(:,2),... % [-X-],[-Y-]
      'Color',randColor,'LineWidth',1.5);
end
hold off;

% Plot First 2 Image Matches
post.matchesI = struct('queryIdx',num2cell(post.matchesInliers(:,1)'),...
   'trainIdx',num2cell(post.matchesInliers(:,2)'),...
   'imgIdx',num2cell(post.matchesInliers(:,3)'),...
   'distance',num2cell(post.matchesInliers(:,4)'));
post.imMatches2 = cv.drawMatches(post.im1.im, post.im1.keypoints,...
   post.im2.im, post.im2.keypoints, post.matchesI);
figure(2); subplot(2,2,1:2); 
% figure();
imshow(post.imMatches2);
title('Matches from First 2 Frames');

% RGB-D Visualization
figure(2); subplot(2,2,3);
% figure(); 
imshow(post.im1.im); hold on;
for i = 1:length(im1.normDepth)
   d = im1.normDepth(i);
   scatter(post.imPointsMatches_12(1,i,1),post.imPointsMatches_12(2,i,1),...
      60,[d/2,d,(1-d)],'Fill');
%    text(post.imPointsMatches_12(1,i,1)+2,post.imPointsMatches_12(2,i,1)-2,...
%       num2str(post.imPointsMatches_12(4,i)),'Color',[1,0.5,0]);
end
title('Image 1 Feature Depth');
hold off;
figure(2); subplot(2,2,4);
% figure(); 
imshow(post.im2.im); hold on;
for i = 1:length(im2.normDepth)
   d = im2.normDepth(i);
   scatter(post.imPointsMatches_12(1,i,2),post.imPointsMatches_12(2,i,2),...
      60,[d/2,d,(1-d)],'Fill');
%    text(post.imPointsMatches_12(1,i,2)+2,post.imPointsMatches_12(2,i,2)-2,...
%       num2str(post.imPointsMatches_12(4,i)),'Color',[1,0.5,0]);
end
title('Image 2 Feature Depth');
hold off;


%% Save a Figure

% saveFigures([1],{'test'},'pdf')

