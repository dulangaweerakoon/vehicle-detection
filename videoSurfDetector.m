reader = vision.VideoFileReader('C:\Users\dulangakw\Dropbox\SMU\BeachStation09042016.avi');
videoPlayer = vision.DeployableVideoPlayer();
%figure
oldframe = rgb2gray(reader.step());
frame = oldframe;

while ~isDone(reader)
    oldframe = frame;
    frame = reader.step();
    frame = rgb2gray(frame);
    
    obj  = detectSURFFeatures(frame);
    
    
    [features, pts] = extractFeatures(frame, obj);
   
   
    %imshow(frame);
    %hold on
    %plot(pts.selectStrongest(100))
    a = pts.selectStrongest(50);
    position = [a.Location,5*ones(50,1)];
    label = ones(1,50);
    
    RGB = insertObjectAnnotation(frame, 'circle', position, label, ...
      'LineWidth', 3,  'TextColor', 'black');
  videoPlayer.step(RGB);
end