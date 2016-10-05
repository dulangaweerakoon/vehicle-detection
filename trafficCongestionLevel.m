close all
load('BW2.mat','BW');
load('threshold.mat','threshold');
I = imread('23.jpg');
%I= I.*BW;
I =rgb2gray(I);
C = corner(I,300);
noOfCorners = size(C);
noOfCorners = noOfCorners(1);
r = []; 
c = [];
for i=1:noOfCorners
    if (BW(C(i,2),C(i,1))==1)
        r = [r; C(i,2)];
        c = [c; C(i,1)];
    end 
end

%cr = [];
%cc = [];
noOfCorners = length(r);
%isAllocated = zeros(1,noOfCorners);
[height,width,~] = size(I);
conerMat = zeros(height,width);

[X,Y] =  meshgrid(1:width,1:height);

for i = 1:length(r)
    
    %conerMat= conerMat==1 | sqrt((X-c(i)).^2 + (Y-r(i)).^2) < 10;
    %mask= mask + (sqrt((X-c(i)).^2 + (Y-r(i)).^2) < rad(i));
    
      conerMat = conerMat + (((X-c(i)).^2 + (Y-r(i)).^2) < 100);
end
sum(sum(BW>0))
congestionLevel = (sum(sum(conerMat>0))./sum(sum(BW==1))).*100
imshow(I)

%conerMat(r,c) = 100;
%H = vision.LocalMaximaFinder('MaximumNumLocalMaxima',10);
%idx = step(H,conerMat);

% conerMat = imdilate(conerMat,[1 1 1; 1 0 1; 1 1 1]);
% [rrr,ccc] = find(conerMat==1);
% I(r,c)=0;
% 
% imshow(I)

% for i=1:noOfCorners
%     if (isAllocated(i) ~= 1)
%         isAllocated(i) = 1;
%         count = 1;
%         cenR = r(i); cenC = c(i);
%         for j=1:noOfCorners
%             if (i~=j)
%                 distance = (r(i)-r(j))^2+(c(i)-c(j))^2
%                 if distance<threshold(ceil((r(j)+r(i))/2))  %3000  
%                     isAllocated(j) = 1;
%                     count = count + 1;
%                     cenR = cenR + r(j);
%                     cenC = cenC + c(j);
%                 end
%             end
%         end
%         cenR = cenR /count;
%         cenC = cenC /count;
%         cr = [cr; cenR];
%         cc = [cc; cenC];
%     end
%     
% end 

 %radii =25*ones(1,length(idx));
% centers = [cc,cr];
%  imshow(I);
%  hold on
%  plot(idx(:,1),idx(:,2), 'r*');
 %viscircles(idx,radii);