I = imread('ppl2.jpg');

%BW = roipoly
ROI = roipoly(I)

save('ROI.mat','ROI');
%csvwrite('BW2.csv',BW)
