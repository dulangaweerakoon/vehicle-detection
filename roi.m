I = imread('wood11.jpg');

%BW = roipoly
BW = roipoly(I)

%save('BW4.mat','BW');
csvwrite('BW2.csv',BW)
