%% Scale invariant blob detector solution code
%% Adapted by Svetlana Lazebnik and Josef Sivic based on code by Serhat Tekin 
%% UNC Chapel Hill, COMP 776 (Spring 2008)
%% ENS Paris, Object recognition and Computer Vision (Fall 2009)
%%
%% Usage:  [r, c, rad] = BlobDetector(im, sigma, k, sigma_final, threshold)
%%
%% Arguments:
%%            im     - input image
%%            sigma  - initial scale
%%            k   - scale multiplication constant
%%            sigma_final - largest scale to process
%%            threshold - Laplacian threshold
%%
%% Returns:
%%            r      - row coordinates of blob centers
%%            c      - column coordinates of blob centers
%%            rad    - circular blob radius
%%
function [r, c, rad] = Blobs(im, sigma, k, sigma_final, threshold)
close all;
load('BW.mat','BW');
if size(im,3)>1
    im = mean(im,3)/255;
   
end;

n = ceil((log(sigma_final) - log(sigma))/log(k)); % number of scale iterations

% allocate state space
[h, w] = size(im); % h, w => height and width of the state space
scaleSpace = zeros(h, w, n);

tic
% generate the Laplacian of Gaussian for the first scale level
filt_size = 2*ceil(3*sigma)+1;  % important: to avoid "shifting" artifacts, make sure the kernel size is odd!
LoG =  sigma^2 * fspecial('log', filt_size, sigma);

% generate the responses for the remaining levels
if 1 % Faster version: keep the filter size, downsample the image
fprintf('Filtering with Laplacian (keep filter size the same, downsample image)... \n');
imRes = im;
for i = 1:n
    fprintf('Sigma %f\n', sigma * k^(i-1));
    imFiltered = imfilter(imRes, LoG, 'same', 'replicate'); % filter the image with LoG
    % note that no scale normalization is needed: the fact that the filter
    % remains the same size while the image is downsampled ensures that the
    % response of the filter is scale-invariant
    imFiltered = imFiltered .^ 2; % save square of the response for current level
    
    % upsample the LoG response to the original image size
    scaleSpace(:,:,i) = imresize(imFiltered, size(im), 'bicubic'); % bilinear supersampling will result in a loss of spatial resolution
    if i < n        
        imRes = imresize(im, 1/(k^i), 'bicubic');
    end
end
toc
end;

% Slower version: increse filter size, keep image the same
if 0 
    fprintf('Filtering with Laplacian keep the image the same, change the filter size...\n');
    scaleSpace2 = zeros(h, w, n);
    for i = 1:n
        sigmai = sigma * k^(i-1);
        fprintf('%d/%d Sigma %f\n', i,n,sigmai);
        
        % generate the Laplacian of Gaussian for the first scale level
        filt_size = 2*ceil(3*sigmai)+1;  % important: to avoid "shifting" artifacts, make sure the kernel size is odd!
        LoG       =  sigmai^2 * fspecial('log', filt_size, sigmai); % scale normalized Laplacian
        
        imFiltered = imfilter(im, LoG, 'same', 'replicate'); % filter the image with LoG
        imFiltered        = imFiltered .^ 2; % square of the response for current level
        scaleSpace2(:,:,i) = imFiltered;     % save response to a 3D array
        
        figure(1); clf; imagesc(imFiltered); colorbar; % show the response
        drawnow;
        pause(0.01);
    end
    toc
    scaleSpace = scaleSpace2;
end;


tic
fprintf('Performing nonmaximum suppression within scales...\n');
% perform non-maximum suppression for each scale-space slice
supprSize = 3;
maxSpace = zeros(h, w, n);
for i = 1:n
    maxSpace(:,:,i) = ordfilt2(scaleSpace(:,:,i), supprSize^2, ones(supprSize));
%    maxSpace(:,:,i) = colfilt(scaleSpace(:,:,i), [supprSize supprSize], 'sliding', @max);
%    % this is a slightly less efficient option
end
toc

% non-maximum suppression between scales and threshold
tic
fprintf('Performing nonmaximum suppression between scales...\n');
for i = 1:n
    maxSpace(:,:,i) = max(maxSpace(:,:,max(i-1,1):min(i+1,n)),[],3);
end
maxSpace = maxSpace .* (maxSpace == scaleSpace);
toc

r = [];   
c = [];   
rad = [];
for i=1:n
    [rows, cols] = find(maxSpace(:,:,i) >= threshold);
    radii =  sigma * k^(i-1) * sqrt(2)*1.5;
    numBlobs = length(rows);
    for j=1:numBlobs
        if (BW(rows(j),cols(j))==1)
             
            %radii = repmat(radii, numBlobs, 1);
            r = [r; rows(j)];
            c = [c; cols(j)];
            rad = [rad; radii];
        end
    end 
%     radii =  sigma * k^(i-1) * sqrt(2); 
%     radii = repmat(radii, numBlobs, 1);
%     r = [r; rows];
%     c = [c; cols];
%     rad = [rad; radii];
end

figure(2); clf;
show_all_circles(im, c, r, rad, 'r', 1.5);
%findIndices(im,r,c);
pause(.1);
drawnow;
figure
getmask(im,c,r,rad);


function show_all_circles(I, cx, cy, rad, color, ln_wid)
%% I: image on top of which you want to display the circles
%% cx, cy: column vectors with x and y coordinates of circle centers
%% rad: column vector with radii of circles. 
%% The sizes of cx, cy, and rad must all be the same
%% color: optional parameter specifying the color of the circles
%%        to be displayed (red by default)
%% ln_wid: line width of circles (optional, 1.5 by default

load('BW.mat','BW');
if nargin < 5
    color = 'r';
end

if nargin < 6
   ln_wid = 1.5;
end

imshow(I.*BW); hold on;

theta = 0:0.1:(2*pi+0.1);
cx1 = cx(:,ones(size(theta)));
cy1 = cy(:,ones(size(theta)));
rad1 = rad(:,ones(size(theta)));
theta = theta(ones(size(cx1,1),1),:);
X = cx1+cos(theta).*rad1;
Y = cy1+sin(theta).*rad1;
line(X', Y', 'Color', color, 'LineWidth', ln_wid);

title(sprintf('%d circles', size(cx,1)));

function mask = getmask(im,c,r,rad)
[h,w,~] = size(im);
[X,Y] = meshgrid(1:w,1:h);
len = length(rad);
mask =zeros(h,w);

for i = 1:len
    
    mask= mask==1 | sqrt((X-c(i)).^2 + (Y-r(i)).^2) < rad(i);
    %mask= mask + (sqrt((X-c(i)).^2 + (Y-r(i)).^2) < rad(i));
    
    %   mask = mask + sqrt((X-c(i)).^2 + (Y-r(i)).^2) < rad(i);
end
%regmax =imregionalmax(mask,8);

mask =imclose(mask,strel('disk',13)); %15
%mask =imdilate(regmax,[1 1 1,1 0 1,1 1 1]);

stat = regionprops('table',mask,'centroid','area');
areas = stat.Area;
centers = stat.Centroid;
inds = find(areas<100); %100
areas(inds) =[];
centers(inds,:)=[];
% for i=1:length(areas)
%     if areas(i)<500
%         areas(i)=[];
%         centers(i)=[]
%     end
% end
radii = 25*ones(1,length(areas));


load('BW.mat','BW');
%mask = mask.*BW;  ##Check 

%imshow(regmax.*BW+im);
%figure
imshow(im);

hold on
viscircles(centers,radii);
hold off;

figure
imshow(mask);
% function indices = findIndices(im,r,c)
%  len = length(r);
%  threshold = 1;
%  clusters = zeros(1,len);
%  for i =1:len
%      for j = 1:len 
%         if (i~=j)
%            distance = sqrt((r(i)-r(j))^2+(c(i)-c(j))^2);
%            if (distance<threshold)
%                 clusters(i) = clusters(i)+1;
%            end 
%         end 
%      end
%  end 
%  
% [~,locs]=findpeaks(clusters);
% figure
% imshow(im); hold on;
% theta = 0:0.1:(2*pi+0.1); 
% r1 = r(:,ones(size(theta)));
% c1 = c(:,ones(size(theta)));
% theta = theta(ones(size(c1,1),1),:);
% X = c1+cos(theta).*10;
% Y = r1+sin(theta).*10;
% line(X', Y', 'Color', 'b', 'LineWidth', 1.5);
% 
%  save('clusters.mat','clusters');