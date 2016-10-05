clear all
clc
I = imread('19.jpg');
BW = im2bw(I, graythresh(I));
[B,L] = bwboundaries(BW,'noholes');
imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on
for k = 1:length(B)
    boundary = B{k};
    plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
end
% I2 = imread('15.jpg');
% % background = imopen(I,strel('disk',15));
% % Ip = imsubtract(I,background);
% % imshow(Ip,[])
% Ic = imsubtract(I,I2);
% imshow(Ic,[]);