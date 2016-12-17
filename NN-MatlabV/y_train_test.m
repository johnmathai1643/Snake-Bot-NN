Y1 = [];
k = 1; 
for i = 1:1:400
 if k == 41
  k = 1;
 elseif mod(i,10) == 0
  Y1 = [Y1;k]; 
  k = k + 1;
 else
  Y1 = [Y1;k];
 end
end

Y_TRAIN = [];
for i = 1:10:391
Y_TRAIN = [Y_TRAIN;Y1(i:i+4,:)];
end

Y_TEST = [];
for i = 10:10:400
Y_TEST = [Y_TEST;Y1(i-4:i,:)];
end