close all;
load('BW.mat','BW');

I = imread('5.jpg');

background = imread('background.jpg');
I(:,:,1) =int16(I(:,:,1)).*int16(BW);
I(:,:,2) =int16(I(:,:,2)).*int16(BW);
I(:,:,3) =int16(I(:,:,3)).*int16(BW);

background(:,:,1) =int16(background(:,:,1)).*int16(BW);
background(:,:,2) =int16(background(:,:,2)).*int16(BW);
background(:,:,3) =int16(background(:,:,3)).*int16(BW);

I = HistNorm(background,I);

I = rgb2gray(I);
background = rgb2gray(background);
%background = imopen(background,strel('disk',3));

Ip = imsubtract(I,background);
%Ip = rgb2gray(Ip);
Ip = imclose(Ip,strel('disk',2));

Ip(Ip<5) =0;
Ip(Ip>=5) =1;

imshow(Ip.*I)
figure
imshow(I)

