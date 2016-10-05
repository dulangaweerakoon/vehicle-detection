threshold = zeros(1,480);

for i = 1:480
    %(1000+2000*exp(-0.005*i))
    if i>150
       threshold(i) = (100+5000*(1-exp(-0.003*(i-150)))); %0.005
    end 
    %threshold(i) = (500+2500*exp(-0.005*i));
end 

save('threshold.mat','threshold');