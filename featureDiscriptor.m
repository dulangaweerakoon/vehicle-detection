close all;
I =imread('1_3.jpg');
I =rgb2gray(I);
obj  = detectSURFFeatures(I);
[features, pts] = extractFeatures(I, obj);

imshow(I)
hold on
plot(pts.selectStrongest(100));