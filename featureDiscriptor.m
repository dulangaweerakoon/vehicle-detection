close all;
I =imread('9.jpg');
I =rgb2gray(I);
obj  = detectHarrisFeatures(I);
[features, pts] = extractFeatures(I, obj);

imshow(I)
hold on
plot(pts.selectStrongest(300));