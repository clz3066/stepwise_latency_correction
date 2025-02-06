%% readin data of each subject and each condition from one folder; then do RIDE

% clear;
load("./channel.mat");
original_path       = '../../../Dataset/face-cognition-data/mat/';
step0_path          = '../../../Dataset/face-cognition-data/RIDE_results/';
step12_path         = '../../../face-cognition-data/step12/';
step12_refined_path = '../../../face-cognition-data/step12_refined/';
step3_path          = '../../../face-cognition-data/step3/';
step4_path          = '../../../face-cognition-data/step4/';
%% Step1: RIDE
files = dir(fullfile(original_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);

for k = 1: len_names
    trace = strcat(original_path, filenames(k));
    [filepath, name, ext] = fileparts(trace);
    contents = split(name, '_');
    if contents(2) == "Fsp"
        if (contents(3) == "PF" || contents(3) == "UPF")

            save_stage0_name = step0_path + convertCharsToStrings(name) + '.mat';
            if ~exist(save_stage0_name, 'file')
                eval(['variables', '=', 'load(trace{1,1})', ';']);
                
                data = variables.tensor;
                data = permute(data, [2, 1, 3]);
                rt = single(variables.rt);
            
                cfg = [];
                cfg.samp_interval = 4;
                cfg.epoch_twd = [-100, 1500];
                cfg.comp.name = {'s', 'c', 'r'};
                cfg.comp.twd = {[0,400], [200,900], [-300,300]};
                cfg.comp.latency = {0, 'unknown', rt};
                cfg = RIDE_cfg(cfg);
                results0 = RIDE_call_new(data, cfg); % run RIDE
                save(save_stage0_name, 'results0');
                disp(k)
            end
        end
        disp(contents)
    end
end
%% Step2: synchronize per condition

files = dir(fullfile(step0_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);
sample_interval = 4;
for k = 1: len_names
    raw_trace = strcat(original_path, filenames(k));
    trace = strcat(step0_path, filenames(k));
    [filepath, name, ext] = fileparts(trace);
    save_stage2_name = step12_path + convertCharsToStrings(name) + '.mat';
    if ~exist(save_stage2_name, 'file')
        eval(['after_RIDE', '=', 'load(trace{1,1})', ';']);
        eval(['raw', '=', 'load(raw_trace{1,1})', ';']);

        data = permute(raw.tensor, [2, 1, 3]);
        trial_num = size(data,3);
        
        results12.st_s = data - ...
            move3(after_RIDE.results0.c(:,:,ones(1,trial_num)), round((after_RIDE.results0.latency_c)/sample_interval)) - ...
            move3(after_RIDE.results0.r(:,:,ones(1,trial_num)), round((after_RIDE.results0.latency_r)/sample_interval));
        results12.st_c = data - after_RIDE.results0.s(:,:,ones(1,trial_num)) - ...
            move3(after_RIDE.results0.r(:,:,ones(1,trial_num)), round((after_RIDE.results0.latency_r)/sample_interval));
        results12.st_r = data - after_RIDE.results0.s(:,:,ones(1,trial_num)) - ...
            move3(after_RIDE.results0.c(:,:,ones(1,trial_num)), round((after_RIDE.results0.latency_c)/sample_interval));

        % synchronize C components within subject per condition
        results12.st_c_synced = move3(results12.st_c, -round(after_RIDE.results0.latency_c/sample_interval));
        results12.st_r_synced = move3(results12.st_r, -round(after_RIDE.results0.latency_r/sample_interval));
        results12.latency_c_idx = round(after_RIDE.results0.latency_c/sample_interval);
        results12.ref_c = after_RIDE.results0.c;     
        
        save(save_stage2_name, 'results12');
    end
end


%% plot template C component and the peak 
%  then refine step12 max_idx results
load("channel.mat");
files = dir(fullfile(step12_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);
for k = 1: len_names
    
    trace = strcat(filtered_path, filenames(k));
    [filepath, name, ext] = fileparts(trace);
    
    load_step12_name   = sprintf('../../../face-cognition-data/step12/%s.mat', convertCharsToStrings(name));
    save_file_name     = sprintf('../../../face-cognition-data/step12_refined/%s.mat', convertCharsToStrings(name));
    save_fig_name      = sprintf('../../../face-cognition-data/filtered/%s.jpg', convertCharsToStrings(name));
    if ~exist(save_fig_name, 'file')
        load(trace{1,1});
        load(load_step12_name);

        t_axis = linspace(1, 400, 400);
        data = results12.ref_c(:, 28);
        % data = data.';
        [pks, locs] = findpeaks(data(75:250), 'minpeakdistance', 20); 
        [max_peak, idx] = max(pks);
        max_idx = locs(idx) + 74;
        results12.ref_c_peak = max_idx;
        save(save_file_name, 'results12');

        f = figure('visible','off');
        plot(t_axis, data, max_idx, max_peak, 'o');
        xticks([25, 75, 125, 175, 225, 275, 325, 375]);
        xticklabels({'0', '200', '400', '600', '800', '1000', '1200', '1400'});
        saveas(f, save_fig_name, 'jpg')
    end
end

%% check the figure and peak, if not appropriate, filter and refind. 
load("channel.mat");
filtered_outlier_path = '../../../face-cognition-data/filtered_outlier/';
files = dir(fullfile(filtered_outlier_path, '*.jpg'));
filenames = {files.name}';
len_names = size(filenames, 1);
for k = 1: len_names
    
    trace = strcat(filtered_outlier_path, filenames(k));
    [filepath, name, ext] = fileparts(trace);
    
    load_step12_name   = sprintf('../../../face-cognition-data/step12/%s.mat', convertCharsToStrings(name));
    save_file_name     = sprintf('../../../face-cognition-data/step12_refined/%s.mat', convertCharsToStrings(name));
    save_fig_name      = sprintf('../../../face-cognition-data/filtered_new_for_outlier/%s.jpg', convertCharsToStrings(name));
    if ~exist(save_fig_name, 'file')
        load(trace{1,1});
        load(load_step12_name);

        t_axis = linspace(1, 400, 400);
        data = results12.ref_c(:, 28);
        [pks, locs] = findpeaks(data(75:150), 'minpeakdistance', 20); 
        [max_peak, idx] = max(pks);
        max_idx = locs(idx) + 74;
        results12.ref_c_peak = max_idx;
        save(save_file_name, 'results12');

        f = figure('visible','off');
        plot(t_axis, data, max_idx, max_peak, 'o');
        xticks([25, 75, 125, 175, 225, 275, 325, 375]);
        xticklabels({'0', '200', '400', '600', '800', '1000', '1200', '1400'});
        saveas(f, save_fig_name, 'jpg')
    end
end


%% Step3: align within the subject
files = dir(fullfile(step12_refined_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);

latency = [];
disp('start synchronize within subject')

for k = 1: len_names
    trace = strcat(step12_refined_path, filenames(k));
    [filepath, name, ext] = fileparts(trace);
    save_name = step3_path + convertCharsToStrings(name) + '.mat';
    outcomes = split(name, '_');
    if outcomes{2} == "Fsp"
        if ~exist(save_name, 'file')
            
            eval(['one', '=', 'load(trace{1,1})', ';']);
        
            if outcomes{3} == "PF"
                anti_task_name = [outcomes{1}, '_', outcomes{2}, '_UPF_cc.mat'];
            elseif outcomes{3}  == "UPF"
                anti_task_name = [outcomes{1}, '_', outcomes{2}, '_PF_cc.mat'];
            end
            
            another_trace = strcat(step12_refined_path, anti_task_name);
            eval(['another', '=', 'load(another_trace)', ';']);
        
            one_corrected_latency_idx = one.results12.ref_c_peak + one.results12.latency_c_idx;
            another_corrected_latency_idx = another.results12.ref_c_peak + another.results12.latency_c_idx;
        
            new_median = median([one_corrected_latency_idx; another_corrected_latency_idx]);
            % new_median = median([one.results12.ref_c_peak, another.results12.ref_c_peak]);
            disp([one.results12.ref_c_peak, another.results12.ref_c_peak, new_median])
            
            one_latency = one_corrected_latency_idx - new_median; 
            another_latency = another_corrected_latency_idx - new_median;
    
            latency = [latency, one_corrected_latency_idx', another_corrected_latency_idx'];
                
            output = {};
            output.median_within_subject = new_median;
            output.st_c_synced_within_subject = move3(one.results12.st_c, -round(one_latency));
            save(save_name, 'output');
            
            output = {};
            output.median_within_subject = new_median;
            output.st_c_synced_within_subject = move3(another.results12.st_c, -round(another_latency));
            save_name = [step3_path, anti_task_name];
            save(save_name, 'output');
        end
    end
end

disp(median(latency))   % Facc: 161  Fsp: 159
%% Step4: align between subjects

files = dir(fullfile(step12_refined_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);

disp('start synchronize between subjects')
for k = 1: len_names
    disp(k)
    trace = strcat(step12_refined_path, filenames(k));
    [filepath, name, ext] = fileparts(trace);
    eval(['one', '=', 'load(trace{1,1})', ';']);

    outcomes = split(name, '_');
    
    if outcomes{2} == "Hacc" && ismember(outcomes{3}, ["PF", "UPF"])
        new_median = 160;
    elseif outcomes{2} == "Facc" && ismember(outcomes{3}, ["PF", "UPF"])
        new_median = 161;
    elseif outcomes{2} == "Hsp" && ismember(outcomes{3}, ["PF", "UPF"])
        new_median = 161;
    elseif outcomes{2} == "Fsp" && ismember(outcomes{3}, ["PF", "UPF"])
        new_median = 159;
    else
        disp('new median is wrong')
    end

    one_corrected_latency_idx = one.results12.ref_c_peak + one.results12.latency_c_idx;
    one_latency = one_corrected_latency_idx - new_median; 

    output = {};
    output.st_c_synced_between_subjects = move3(one.results12.st_c, -round(one_latency));
    save_name = [step4_path, name, '.mat'];
    save(save_name, 'output');
end
disp('finished')

%% aligned amplitude
% the code is in 'stage4.py' and data saved in 'gfp_step4';

%% Merge: combine data from all stages
clear;

original_path         = '../../../face-cognition-data/mat/';
step0_path            = '../../../face-cognition-data/RIDE_results/';
step12_refined_path   = '../../../face-cognition-data/step12_refined/';
step3_path            = '../../../face-cognition-data/step3/';
step34_path           = '../../../face-cognition-data/step4/';
step4_path            = '../../../face-cognition-data/gfp_step4/';
save_path             = '../../../face-cognition-data/save_all_stages/';

files = dir(fullfile(step3_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);

disp('start combine data from all stages')
for k = 1: len_names
    disp(k)
    output = {};
    trace = strcat(original_path, filenames(k));
    original = load(trace{1,1});
    trace = strcat(step0_path, filenames(k));
    step0 = load(trace{1,1});    
    trace = strcat(step12_refined_path, filenames(k));
    step12 = load(trace{1,1});  
    trace = strcat(step3_path, filenames(k));
    step3 = load(trace{1,1});
    trace = strcat(step34_path, filenames(k));
    step34 = load(trace{1,1});       
    trace = strcat(step4_path, filenames(k));
    step4 = load(trace{1,1});
    results.aligned_amp = step4.aligned_amp;

    tensor = permute(original.tensor, [2, 1, 3]);
    rt = double(original.rt);
    results.original = tensor;
    results.rt = rt;

    names = fieldnames(step0.results0); % 获取mat中所有变量的名字
    for j = 1: size(names, 1)
        eval(['results.', names{j}, '= step0.results0.(names{j});']);
    end

    names = fieldnames(step12.results12); % 获取mat中所有变量的名字
    for j = 1: size(names, 1)
        % name = step0.results.(names{k}); % 取出第一个变量
        eval(['results.', names{j}, '= step12.results12.(names{j});']);
    end

    names = fieldnames(step3.output); % 获取mat中所有变量的名字
    for j = 1: size(names, 1)
        % name = step0.results.(names{k}); % 取出第一个变量
        eval(['results.', names{j}, '= step3.output.(names{j});']);
    end

    names = fieldnames(step34.output); % 获取mat中所有变量的名字
    for j = 1: size(names, 1)
        % name = step0.results.(names{k}); % 取出第一个变量
        eval(['results.', names{j}, '= step34.output.(names{j});']);
    end

 
    save_name = strcat(save_path, filenames(k));
    save(save_name{1,1}, "results");

end
disp('finished all combining')


%% plot the peak of ERP_new or st_c_synced (directly find the peak)
%  then refine step12 max_idx results
load("channel.mat");
file_name = 'erp_new';
filtered_path = sprintf('../../../Dataset/filtered/%s/', file_name);
files = dir(fullfile(filtered_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);
for k = 1: len_names
    
    trace = strcat(filtered_path, filenames(k));
    [filepath, name, ext] = fileparts(trace);
    save_fig_name      = sprintf('../../../Dataset/filtered/%s_fig/%s.jpg', file_name, convertCharsToStrings(name));
    % load_step12_name   = sprintf('../../../Dataset/step12/%s.mat', convertCharsToStrings(name));
    save_file_name     = sprintf('../../../Dataset/filtered/%s_refined/%s.mat', file_name, convertCharsToStrings(name));
    if ~exist(save_file_name, 'file')
        results = load(trace{1,1});
        % load(load_step12_name);

        t_axis = linspace(1, 400, 400);
        data = results.data.';
        [pks, locs] = findpeaks(data(40:225), 'minpeakdistance', 20); 
        [max_peak, idx]  = max(pks);
        max_idx = locs(idx) + 39;
        results.peak = max_idx;
        results.amp = data(max_idx);
        save(save_file_name, 'results');
        disp(max_idx);

        % f = figure('visible','off');
        % plot(t_axis, data, max_idx, max_peak, 'o');
        % xticks([25, 75, 125, 175, 225, 275, 325, 375]);
        % xticklabels({'0', '200', '400', '600', '800', '1000', '1200', '1400'});
        % saveas(f, save_fig_name, 'jpg')
    end
end


%% plot the peak of ERP_new or st_c_synced (using cross covariance/cross correlation)
%  then refine step12 max_idx results
load("channel.mat");
file_name = 'st_c_synced';
filtered_path = sprintf('../../../Dataset/filtered/%s/', file_name);
files = dir(fullfile(filtered_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);

for k = 1: len_names

    trace = strcat(filtered_path, filenames(k));
    results = load(trace{1,1});
    [filepath, name, ext] = fileparts(trace);
    outcomes = split(name, '_');
    if outcomes{2} == "Facc" && outcomes{3} == "PF"
        load('../ERP_results/ERP/Facc_step2.mat');
        Pz = p(28,:);
    elseif outcomes{2} == "Facc" && outcomes{3} == "UPF"
        load('../ERP_results/ERP/Facc_step2.mat');
        Pz = up(28,:);
    elseif outcomes{2} == "Fsp" && outcomes{3} == "PF"
        load('../ERP_results/ERP/Fsp_step2.mat');
        Pz = p(28,:);
    elseif outcomes{2} == "Fsp" && outcomes{3} == "UPF"
        load('../ERP_results/ERP/Fsp_step2.mat');
        Pz = up(28,:);
    end

    t_axis = linspace(1, 400, 400);
    data = results.data.';
    
    [cross_corr, lags] = xcorr(data, Pz, 200);
    [pks, locs] = findpeaks(cross_corr(1:400), 'minpeakdistance', 20); 
    [max_peak, idx]  = max(pks);
    max_idx = locs(idx);

    % f = figure('visible','on');
    % plot(lags, cross_corr, lags(max_idx), max_peak, 'o');
    results.ref_c_peak = lags(max_idx);    
    disp(lags(max_idx));

    save_file_name = sprintf('../../../Dataset/filtered/%s_refined/%s.mat', file_name, convertCharsToStrings(name));
    save(save_file_name, 'results');
    clear Pz;
end



%% plot the peak of each subject 
%  then refine step12 max_idx results
load("channel.mat");
file_name = 'subject';
filtered_path = '../../../GitHub/priming/ERP_results/subject_ERP/';
files = dir(fullfile(filtered_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);
for k = 1: len_names
    
    trace = strcat(filtered_path, filenames(k));
    [filepath, name, ext] = fileparts(trace);
    save_fig_name      = sprintf('../../../Dataset/filtered/%s_fig/%s.jpg', file_name, convertCharsToStrings(name));
    % load_step12_name   = sprintf('../../../Dataset/step12/%s.mat', convertCharsToStrings(name));
    save_file_name     = sprintf('../../../Dataset/filtered/%s_refined/%s.mat', file_name, convertCharsToStrings(name));
    if ~exist(save_file_name, 'file')
        results = load(trace{1,1});
        % load(load_step12_name);

        t_axis = linspace(1, 400, 400);

        data = results.erp_step0.';
        [pks, locs] = findpeaks(data(75:175, 28), 'minpeakdistance', 20); 
        [max_peak, idx]  = max(pks);
        max_idx = locs(idx) + 74;
        results.peak0 = max_idx;
        results.amp0 = data(max_idx, 28);
        disp(max_idx);

        data = results.erp_step1.';
        [pks, locs] = findpeaks(data(75:175, 28), 'minpeakdistance', 20); 
        [max_peak, idx]  = max(pks);
        max_idx = locs(idx) + 74;
        results.peak1 = max_idx;
        results.amp1 = data(max_idx, 28);
        disp(max_idx);

        data = results.erp_step2.';
        [pks, locs] = findpeaks(data(75:175, 28), 'minpeakdistance', 20); 
        [max_peak, idx]  = max(pks);
        max_idx = locs(idx) + 74;
        results.peak2 = max_idx;
        results.amp2 = data(max_idx, 28);
        disp(max_idx);

        data = results.erp_step3.';
        [pks, locs] = findpeaks(data(75:175, 28), 'minpeakdistance', 20); 
        [max_peak, idx]  = max(pks);
        max_idx = locs(idx) + 74;
        results.peak3 = max_idx;
        results.amp3 = data(max_idx, 28);
        save(save_file_name, 'results');
        disp(max_idx);

        % f = figure('visible','off');
        % plot(t_axis, data(:, 28), max_idx, max_peak, 'o');
        % xticks([25, 75, 125, 175, 225, 275, 325, 375]);
        % xticklabels({'0', '200', '400', '600', '800', '1000', '1200', '1400'});
        % saveas(f, save_fig_name, 'jpg')

    end
end

%% 
original_path         = '../../../Dataset/Facc_Fsp/';
step4_path            = '../../../Dataset/gfp_step4/';
save_path             = '../../../Dataset/new_all/';

files = dir(fullfile(step4_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);

disp('start combine data from all stages')
for k = 1: len_names
    output = {};
    trace = strcat(original_path, filenames(k));
    original = load(trace{1,1});
    trace = strcat(step4_path, filenames(k));
    step4 = load(trace{1,1});       


    names = fieldnames(original.results); % 获取mat中所有变量的名字
    for j = 1: size(names, 1)
        eval(['results.', names{j}, '= original.results.(names{j});']);
    end

    results.aligned_amp = step4.aligned_amp;

    save_name = strcat(save_path, filenames(k));
    save(save_name{1,1}, "results");

end
disp('finished all combining')