

%% for synchronizing amplitudes between subjects - min-max scaling
input_path     = '../../../save_all_stages/';
output_path    = '../../../amplitudes_synchronized_new/';

files = dir(fullfile(input_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);

disp('start combine data from all stages')
for k = 1: len_names
    trace = strcat(input_path, filenames(k));
    eval(['load(trace{1,1})', ';']);
   
    results.st_c_synced_amp_synced = zeros(size(results.st_c_synced_between_subjects));
    for trial_idx = 1:size(results.st_c_synced, 3)
        single_trial = results.st_c_synced_between_subjects(:,:,trial_idx);

        min_x = min(single_trial); % 每一列的最小值
        max_x = max(single_trial); % 每一列的最大值
        x_normalized = (single_trial - min_x) ./ (max_x - min_x); % 归一化后的 x
        results.st_c_synced_amp_synced(:,:,trial_idx) = x_normalized;
    end
    save_name = strcat(output_path, filenames(k));
    save(save_name{1,1}, "results");
    disp(k)
end


%% for synchronizing amplitudes within subject
load("./channel.mat");
ch_label = 'Pz';
ch_index = find(strcmpi({chanlocs.labels}, ch_label));

input_path     = '../../../Dataset/Facc_Fsp/';
output_path    = '../../../Dataset/amplitudes_synchronized/';

files = dir(fullfile(input_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);

disp('start combine data from all stages')
for k = 1: len_names
    trace = strcat(input_path, filenames(k));
    [filepath, name, ext] = fileparts(trace);
    contents = split(name, '_');
    if contents(2) == "Fsp"
    
        eval(['load(trace{1,1})', ';']);
    
        ref_Pz_amp = 10;
        aligned_amp = zeros(size(results.st_c_synced_between_subjects));
        for trial_idx = 1:size(results.st_c_synced, 3)
            single_trial = results.st_c_synced(:, :, trial_idx);
            single_trial_Pz = single_trial(:, ch_index);
            single_trial_Pz_amp = single_trial_Pz(results.ref_c_peak);
    
            diff = ref_Pz_amp / single_trial_Pz_amp;
            % diff = mean(single_trial_Pz);
            single_trial_between_subjects = results.st_c_synced_between_subjects(:,:,trial_idx);
            syn_single_trial = single_trial_between_subjects * diff;
            aligned_amp(:, :, trial_idx) = syn_single_trial;
    
        end
        save_name = strcat(output_path, filenames(k));
        save(save_name{1,1}, 'aligned_amp');
    end
end

disp(median(amp_lst))

%%
t_axis = linspace(-100, 1500, size(data,1));
[tem1, tem2] = sort(results0.amp_c(:,28));
figure;
imagesc(t_axis, 1:size(data,3), squeeze(st_c_synced_amp_synced(:,28,tem2))');
% colormap(jet);
xlabel('time after stimulus (ms)');
ylabel('trial (sorted)');
title('Single trial ERP sorting by C amplitude from Pz electrode');

%%
figure;
plot(a)
hold on;
plot(b)

%% get S component
step0_path    = '../../../Facc_Fsp/';
s_path        = '../../../S_component/';
files = dir(fullfile(step0_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);
sample_interval = 4;
for k = 1: len_names
    trace = strcat(step0_path, filenames(k));
    [filepath, name, ext] = fileparts(trace);
    save_s_name = s_path + convertCharsToStrings(name) + '.mat';
    if ~exist(save_s_name, 'file')
        eval(['after_RIDE', '=', 'load(trace{1,1})', ';']);

        data = after_RIDE.results.original;

        trial_num = size(data, 3);
        
        s_component = data - ...
            move3(after_RIDE.results.c(:,:,ones(1, trial_num)), round((after_RIDE.results.latency_c)/sample_interval)) - ...
            move3(after_RIDE.results.r(:,:,ones(1, trial_num)), round((after_RIDE.results.latency_r)/sample_interval));
  
        save(save_s_name, 's_component');
        disp(save_s_name)
    end
end

%% 
step0_path    = '../../../Facc_Fsp/';
files = dir(fullfile(step0_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);
Fsp_F_latency = [];

for k = 1: len_names
    disp(k)
    trace = strcat(step0_path, filenames(k));
    [filepath, name, ext] = fileparts(trace);
    save_name = '../../../Facc_Fsp_new/' + convertCharsToStrings(name) +'.mat';
    outcomes = split(name, '_');
    if outcomes{2} == "Fsp"
        if ~exist(save_name, 'file')
            eval(['load(trace{1,1})', ';']);
        
            if outcomes{3} == "PF"
                anti_task_name = [outcomes{1}, '_', outcomes{2}, '_UPF_cc.mat'];
            elseif outcomes{3}  == "UPF"
                anti_task_name = [outcomes{1}, '_', outcomes{2}, '_PF_cc.mat'];
            end
            
            another_trace = strcat(step0_path, anti_task_name);
            eval(['another', '=', 'load(another_trace)', ';']);
        
            one_corrected_latency_idx = results.ref_c_peak + results.latency_c_idx;
            another_corrected_latency_idx = another.results.ref_c_peak + another.results.latency_c_idx;
        
            new_median = median([one_corrected_latency_idx; another_corrected_latency_idx]);
            disp([results.ref_c_peak, another.results.ref_c_peak, new_median])
            one_latency = one_corrected_latency_idx - new_median; 
            another_latency = another_corrected_latency_idx - new_median;
    
            Fsp_F_latency = [Fsp_F_latency, one_corrected_latency_idx'];
                
            results.median_within_subject = new_median;
            results.st_c_synced_within_subject = move3(results.st_c, -round(one_latency));
            save(save_name, 'results');
            
        end
    end
end
% disp(median(Facc_F_latency))  % 163
% disp(median(Fsp_F_latency))   % 159


%% align between subjects
step0_path    = '../../../Facc_Fsp_new/';
files = dir(fullfile(step0_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);
sample_interval = 4;

for k = 1: len_names
    disp(k)
    trace = strcat(step0_path, filenames(k));
    [filepath, name, ext] = fileparts(trace);
    save_name = '../../../Facc_Fsp_new_new/' + convertCharsToStrings(name) +'.mat';
    outcomes = split(name, '_');
    eval(['load(trace{1,1})', ';']);

    if outcomes{2} == "Facc" && ismember(outcomes{3}, ["PF", "UPF"])
        new_median = 163;
    elseif outcomes{2} == "Facc" && ismember(outcomes{3}, ["PUF", "UPUF"])
        new_median = 177;
    elseif outcomes{2} == "Fsp" && ismember(outcomes{3}, ["PF", "UPF"])
        new_median = 159;
    elseif outcomes{2} == "Fsp" && ismember(outcomes{3}, ["PUF", "UPUF"])
        new_median = 172;
    else
        disp('new median is wrong')
    end

    one_corrected_latency_idx = results.ref_c_peak + results.latency_c_idx;
    one_latency = one_corrected_latency_idx - new_median; 

    results.st_c_synced_between_subjects = move3(results.st_c, -round(one_latency));
    save(save_name, 'results');
end
disp('finished')