
%im = imread('9.jpg');
im = imread('1.jpg');
%r=0; %0.8   T =0.07
[r,c,d] = Blobs(im,0.8,2^(0.25),5,0.08);