  load('BW1.mat','BW');
  load('threshold.mat','threshold');
  
  reader = vision.VideoFileReader('video1.avi');
  i =0;
  j=0;
while ~isDone(reader)
    frame = reader.step();
    if i==0
        imwrite(frame,strcat('C:\Group Detection Project\Vignesh BG Substract\BGsubtract\dataset\',int2str(j),'.jpg'))
    end 
    if (i==150)
        i =0;
        j=j+1;
        imwrite(frame,strcat('C:\Group Detection Project\Vignesh BG Substract\BGsubtract\dataset\',int2str(j),'.jpg'))
    end 
    i = i+1;
end 