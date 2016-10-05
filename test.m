close all;

original = imread('C:\Group Detection Project\Vignesh BG Substract\BGsubtract\dataset\1.jpg');
original = rgb2gray(original);
figure;
imshow(original);

scale = 1.3;
J = imresize(original,scale);
theta = 31;
%distorted = imrotate(J,theta);
distorted = imread('C:\Group Detection Project\Vignesh BG Substract\BGsubtract\dataset\2.jpg');
distorted = rgb2gray(distorted);
figure
imshow(distorted)

ptsOriginal  = detectSURFFeatures(original);
ptsDistorted = detectSURFFeatures(distorted);

[featuresOriginal,validPtsOriginal]  = extractFeatures(original,ptsOriginal);
[featuresDistorted,validPtsDistorted] = extractFeatures(distorted,ptsDistorted);

indexPairs = matchFeatures(featuresOriginal,featuresDistorted);

matchedOriginal  = validPtsOriginal(indexPairs(:,1));
matchedDistorted = validPtsDistorted(indexPairs(:,2));

figure
showMatchedFeatures(original,distorted,matchedOriginal,matchedDistorted)
title('Candidate matched points (including outliers)')


%Not Needed..................................................................
% [tform, inlierDistorted,inlierOriginal] = estimateGeometricTransform(matchedDistorted,matchedOriginal,'similarity');
% 
% figure
% showMatchedFeatures(original,distorted,inlierOriginal,inlierDistorted)
% title('Matching points (inliers only)')
% legend('ptsOriginal','ptsDistorted')
% 
% figure
% showMatchedFeatures(original,distorted,inlierOriginal,inlierDistorted)
% title('Matching points (inliers only)')
% legend('ptsOriginal','ptsDistorted')
% 
% outputView = imref2d(size(original));
% recovered  = imwarp(distorted,tform,'OutputView',outputView);
% 
% figure
% imshowpair(original,recovered,'montage')