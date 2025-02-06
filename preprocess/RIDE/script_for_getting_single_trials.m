% clear
%load the sample data (/w RT) for a single subject from a face recognition task
for sec_load_data = 1:1
    % if you don't want to open the data manually you can use this script;
    % load('samp_face.mat');
    load('252014_Fsp_PF_cc.mat')
end
load("./channel.mat");

%configuration of parameters and run RIDE
for section = 1:1
    rt = single(rt);
    data = permute(data, [2, 1, 3]);

    cfg = [];%initialization
    cfg.samp_interval = 4;
    cfg.epoch_twd = [-100, 1500];%time window for the epoched data (relative to stimulus)
    cfg.comp.name = {'s','c','r'}; %component names
    cfg.comp.twd = {[0, 300],[200, 900],[-300, 300]}; %time windows for extracting components, for 's' and 'c' it is raltive to stimulus, for 'r' it is relative to RT
    cfg.comp.latency = {0, 'unknown', rt};%latency for each RIDE component
    
    cfg = RIDE_cfg(cfg);%standardize
    results = RIDE_call_new(data,cfg); %run RIDE
end

%%
%time axis;
t_axis = linspace(cfg.epoch_twd(1), cfg.epoch_twd(2), size(data,1));

%first, let's randomly plot single trials ERP from a selected electrode
ch_label = 'Pz';
ch_index = find(strcmpi({chanlocs.labels}, ch_label));

figure;
imagesc(t_axis,1:size(data,3),squeeze(data(:,ch_index,:))');colormap(jet);
xlabel('time after stimulus (ms)');
ylabel('trial');
title('Single trial ERP without sorting from Pz electrode');

%the figure above was not sorted by C latency or RT so we do not see
%any systematic latency shifting effect. We can plot the same figure
%according to the latency of C (or according to RT)
[tem1, tem2] = sort(results.latency_c);
figure;
imagesc(t_axis, 1:size(data,3), squeeze(data(:,ch_index,tem2))');
colormap(jet);
xlabel('time after stimulus (ms)');
ylabel('trial (sorted)');
title('Single trial ERP sorting by C latency from Pz electrode');

[tem1, tem2] = sort(results.amp_c(:,28));
figure;
imagesc(t_axis, 1:size(data,3), squeeze(data(:,ch_index,tem2))');
colormap(jet);
xlabel('time after stimulus (ms)');
ylabel('trial (sorted)');
title('Single trial ERP sorting by C latency from Pz electrode');
%%
%now we see very systematic latency variability in C, but we don't seem
%to see an S component here, that is because S component is not clear
%in this electrode, if you plot PO8 (but C is not strong there)
ch_label = 'PO8';
ch_index = find(strcmpi({chanlocs.labels},ch_label));

figure;
imagesc(t_axis,1:size(data,3),squeeze(data(:,ch_index,:))');
colormap(jet);
xlabel('time after stimulus (ms)');
ylabel('trial');
title('Single trial ERP without sorting from PO8 electrode');

%in this dataset, R is also strongest at Pz, so if you plot Pz
%according to the accending RT, you also see a latency (RT) depdendent
%pattern:
ch_label = 'Pz';
ch_index = find(strcmpi({chanlocs.labels},ch_label));
[tem1,tem2] = sort(results.latency_r);

figure;
imagesc(t_axis,1:size(data,3),squeeze(data(:,ch_index,tem2))');colormap(jet);
xlabel('time after stimulus (ms)');
ylabel('trial (sorted)');
title('Single trial ERP sorting by R latency from Pz electrode');


%% 
%now, let's try to get the single trial S, C, and R


%'move3' is a function in RIDE. It shifts every single trials from a 3-D (epoch*electrode*trial)
%data with a relative lag
%usuage: data1 = move3(data,lag);
%where data is the 3-D data, and lag is the vector for single trial lags.
trial_num = size(data,3);

ST_S = data - move3(results.c(:,:,ones(1,trial_num)),round(results.latency_c/cfg.samp_interval)) ...
    - move3(results.r(:,:,ones(1,trial_num)),round((results.latency_r)/cfg.samp_interval));

ST_C = data - results.s(:,:,ones(1,trial_num)) - ...
    move3(results.r(:,:,ones(1,trial_num)),round((results.latency_r)/cfg.samp_interval));

ST_R = data - results.s(:,:,ones(1,trial_num)) - ...
    move3(results.c(:,:,ones(1,trial_num)),round(results.latency_c/cfg.samp_interval));


%note that since the latency values provided by RIDE are always in the
%unit of millisecond, we need to convert it back to data unit


%now, ST_S, ST_C, ST_R are just in the same format of data, but free of the other two components

%to validate, let's plot single trial S and single trial C at PO8
%(where S is strong)

%%
ch_label = 'PO8';
ch_index = find(strcmpi({chanlocs.labels},ch_label));

figure;subplot(1,2,1);
imagesc(t_axis,1:size(data,3),squeeze(ST_S(:,ch_index,:))');colormap(jet);
xlabel('time after stimulus (ms)');
ylabel('trial');
title('single trial S');
subplot(1,2,2);
imagesc(t_axis,1:size(data,3),squeeze(ST_C(:,ch_index,:))');colormap(jet);
xlabel('time after stimulus (ms)');
ylabel('trial');
title('single trial C');

%to futher validate C and R is removed at the right latency, there shouldn't be systematic C-latency dependent component in C anymore:
% let's compare single trial data before and after removing C and R:
%time axis;
t_axis = linspace(cfg.epoch_twd(1), cfg.epoch_twd(2), size(data,1));
ch_label = 'Pz';
ch_index = find(strcmpi({chanlocs.labels},ch_label));
[tem1,tem2] = sort(results.latency_c);

figure;subplot(1,2,1);
imagesc(t_axis,1:size(data,3),squeeze(data(:,ch_index,tem2))');colormap(jet);
xlabel('time after stimulus (ms)');
ylabel('trial (sorted)');
title('before removing C and R');
subplot(1,2,2);
imagesc(t_axis,1:size(data,3),squeeze(ST_S(:,ch_index,tem2))');colormap(jet);
xlabel('time after stimulus (ms)');
ylabel('trial (sorted)');
title('after removing C and R');

%%
[tem1, tem2] = sort(results.amp_c(:,28));
figure;
imagesc(t_axis, 1:size(data,3), squeeze(ST_C(:,ch_index,tem2))');
% colormap(jet);
xlabel('time after stimulus (ms)');
ylabel('trial (sorted)');
title('Single trial ERP sorting by C amplitude from Pz electrode');
%%
%last step: let's synchronize C to its single trial latency:
ST_C_synced = move3(ST_C, -round(results.latency_c/cfg.samp_interval));

%Important note: when you sync C, you need to do zero padding, so you
%will see zero values in the single trials


%let compare pre_sync and post_sync C data (sorted by C latency):
ch_label = 'Pz';
ch_index = find(strcmpi({chanlocs.labels},ch_label));
[tem1,tem2] = sort(results.latency_c);

figure;subplot(1,2,1);
imagesc(t_axis,1:size(data,3),squeeze(ST_C(:,ch_index,tem2))');colormap(jet);
xlabel('time after stimulus (ms)');
ylabel('trial (sorted)');
title('single trial C');
subplot(1,2,2);
imagesc(t_axis,1:size(data,3),squeeze(ST_C_synced(:,ch_index,tem2))');colormap(jet);
xlabel('time after stimulus (ms)');
ylabel('trial (sorted)');
title('single trial C (synced)');



%% synchronize p and up
%the above is syncing to its own most probably (median) latency, if you
%have two conditions (say, p, up), you want to sync to a 'common'
%latency between these two conditions, you have to identify a way for
%you to do this kind of syncing. Here, let's say we are refering to the
%peak latency of C at Pz electrode. In the current sample data, the
%peak latency is 489 ms, which is what you can get from the following
%plot.
figure;plot(t_axis,results.c(:,ch_index));xlabel('time after stimulus (ms)');

%%
%assuming that this sample data is primed (p), and there is another
%dataset that is unprimed (up) that has a peak latency of C at Pz at
%554 ms. Therefore, what you want to do maybe to shift the current data
%rightward to the peak latency of (554+489)/2 = 522 ms (rounded). You
%have to shift the single trial C rightwards for 33 ms, and shift the
%other dataset (up) leftwards for 33 ms. To make this shift, just do
%the following:

within = latency_c_idx + ref_c_peak - median_within_subject;
ST_C_synced_con = move3(ST_C_synced, -within);%positive value means shifting rightwards

%do the same thing for up but leftwards

%let compare pre_sync and post_sync C data (sorted by C latency):
ch_label = 'Pz';
ch_index = find(strcmpi({chanlocs.labels},ch_label));
[tem1,tem2] = sort(results.latency_c);

figure;subplot(1,2,1);
imagesc(t_axis,1:size(data,3),squeeze(ST_C_synced(:,ch_index,tem2))');colormap(jet);title('single trial C (synced)');
xlabel('time after stimulus (ms)');
ylabel('trial (sorted)');
subplot(1,2,2);
imagesc(t_axis,1:size(data,3),squeeze(ST_C_synced_con(:,ch_index,tem2))');colormap(jet);title('single trial C (synced) rightwards');
xlabel('time after stimulus (ms)');
ylabel('trial (sorted)');



