reader = vision.VideoFileReader('C:\Users\dulangakw\Dropbox\SMU\BeachStation09042016.avi');
videoPlayer = vision.DeployableVideoPlayer();
%figure
oldframe = rgb2gray(reader.step());
frame = oldframe;

load('ROI.mat','ROI');
while ~isDone(reader)
    oldframe = frame;
    frame = reader.step();
    frame = rgb2gray(frame);
    
    obj  = detectSURFFeatures(frame);
    
    
    [features, pts] = extractFeatures(frame, obj);
   
   
    %imshow(frame);
    %hold on
    %plot(pts.selectStrongest(100))
    cc = pts.selectStrongest(50);
    cc
    locations = cc.Location;
    laplacian = cc.Scale;
    
    position = zeros(0,3);
    label = ones(1,0);
    for i=1:length(locations)
        if ROI(floor(locations(i,2)),floor(locations(i,1)) )
                label = [label ; laplacian(i)];
                position = [position ; [locations(i,:),5]];
                %bboxes = [bboxes;[(cc(i,1)-5),(cc(i,2)-5),10,10]];
        end
    end 
    
    %position = [cc.Location,5*ones(50,1)];
    %label = ones(1,length(position));
    
    RGB = insertObjectAnnotation(frame, 'circle', position, label, ...
      'LineWidth', 3,  'TextColor', 'black');
  videoPlayer.step(RGB);
end