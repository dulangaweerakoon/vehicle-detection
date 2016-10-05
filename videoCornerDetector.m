  load('BW1.mat','BW');
  load('threshold.mat','threshold');
  
  reader = vision.VideoFileReader('video1.avi');
  
  videoPlayer = vision.DeployableVideoPlayer();
  frame1 = reader.step();
  for i =1:10
    frame2 = reader.step();
  end 
  imwrite(frame1,'C:\Group Detection Project\Vignesh BG Substract\BGsubtract\output\frame1.jpg');
  imwrite(frame2,'C:\Group Detection Project\Vignesh BG Substract\BGsubtract\output\frame2.jpg');
  figure 
  hold on 
  while ~isDone(reader)
    I = reader.step();
    I =rgb2gray(I);
    
    C = corner(I,320);
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

    cr = [];
    cc = [];
    noOfCorners = length(r);
    isAllocated = zeros(1,noOfCorners);

    for i=1:noOfCorners
        if (isAllocated(i) ~= 1)
            isAllocated(i) = 1;
            count = 1;
            cenR = r(i); cenC = c(i);
            for j=1:noOfCorners
                if (i~=j)
                    distance = (r(i)-r(j))^2+(c(i)-c(j))^2
                    if distance<threshold(ceil((r(j)+r(i))/2))  %3000  
                        isAllocated(j) = 1;
                        count = count + 1;
                        cenR = cenR + r(j);
                        cenC = cenC + c(j);
                    end
                end
            end
            cenR = cenR /count;
            cenC = cenC /count;
            cr = [cr; cenR];
            cc = [cc; cenC];
        end
    
    end 

    radii =25*ones(1,length(cc));
    centers = [cc,cr];
    for i=1:length(cc)
        I=insertShape(I,'circle',[cc(i),cr(i),radii(i)],'LineWidth',2);
    end 
    %I=insertShape(I,'circle',[50 50 30],'LineWidth',2);
    imshow(I);
    %hold on
    %plot(c, r, 'r*');
    %viscircles(centers,radii);
    
  end 