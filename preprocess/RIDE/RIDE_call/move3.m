function temp = move3(data,latency)
% for j = 1:size(data,3)
%     temp(:,:,j) = move2(data(:,:,j),round(latency(j)),'1d');
% end
% f = temp;
latency = latency(:)';
latency = round(latency);
[d1,d2,d3] = size(data);
temp = zeros(d1,d2,d3);

left = latency+1;left(left<=0) = 1;
right = d1+latency;right(right>d1) = d1;

left1 = -latency+1;left1(left1<=0) = 1;
right1 = d1-latency;right1(right1>d1) = d1;

% original moving: replace by 0
for j = find(latency>-d1&latency<=d1)
    temp(left(j):right(j),:,j) = data(left1(j):right1(j),:,j);
end


% % new moving v1: replace by the mean of each electrode signals
% for idx = find(latency>-d1&latency<=d1)
%     temp(left(idx):right(idx),:,idx) = data(left1(idx):right1(idx),:,idx);
%     if left(idx)==1 && right(idx)<d1               % move left, orignal latency > 0, synchronized latency < 0
%         cut = data(right(idx)+1:400,:,idx);        % pad rightward
%         c1 = size(cut, 1);
%         cut_mean = mean(cut, 1);
%         r = repmat(cut_mean, c1, 1);
%         temp(right(idx)+1:400,:,idx) = r;
%         temp(1:25,:,idx) = data(1:25,:,idx);
%     elseif left(idx)>1 && right(idx)==d1           % move right, orignal latency < 0, synchronized latency > 0
%         cut = data(1:left(idx)-1,:,idx);           % pad leftward
%         c1 = size(cut, 1);
%         cut_mean = mean(cut, 1);
%         r = repmat(cut_mean, c1, 1);
%         temp(1:left(idx)-1,:,idx)= r;
%     end
% end


% % new moving v2: replace by the original signals and cut before onset
% for idx = find(latency>-d1&latency<=d1)
%     temp(left(idx):right(idx),:,idx) = data(left1(idx):right1(idx),:,idx);
%     if left(idx)==1 && right(idx)<d1               % move left, orignal latency > 0, synchronized latency < 0 
%         temp(right(idx)+1:400,:,idx) = data(right(idx)+1:400,:,idx);  % pad right
%         temp(1:25,:,idx) = data(1:25,:,idx);
%     elseif left(idx)>1 && right(idx)==d1           % move right, orignal latency < 0, synchronized latency > 0
%         temp(1:left(idx)-1,:,idx)= data(1:left(idx)-1,:,idx);         % pad left
%         temp(d1-26:d1,:,idx) = data(d1-26:d1,:,idx);
%     end
end










