%% from the original .set file to .mat file
path = '../../../new_dataset/face-cognition-data/face/';
files = dir(fullfile(path, '*.set'));

filenames = {files.name}';
len_names = size(filenames, 1);

for k = 1: len_names
    trace = strcat(path, filenames(k));
    [filepath, name, ext] = fileparts(trace);
    EEG = pop_loadset(trace{1,1});
    EEG = pop_eegfiltnew(EEG, 0, 40, [], 0, 0, 0);
    EEG = pop_resample(EEG, 250);
    data = EEG.data;
    %%4 average reference
    data = data - repmat(mean(data),size(data,1),1);
    tensor = data(:, 26:425, :);  % [-100, 1500]
    disp(name)
    file_mat = char(name(1,:));
    trials = EEG.trials;
    epoch = EEG.epoch;
    rt = [];
    try
        for i=1:trials
            eventtype = epoch(i).eventtype;
            rt_type = find(strcmp(eventtype, 'S254'));
            if isempty(rt_type)
                rt_type = find(strcmp(eventtype, 'S251'));
            end
            eventlatency = epoch(i).eventlatency;
            reaction_time = eventlatency(rt_type);      % extract latency
            rt = [rt, reaction_time{1,1}];
        end

        save(['../../../new_dataset/face-cognition-data/mat/', file_mat, '.mat'], 'tensor', 'rt');
    catch
    end
end

%%
path = '..\..\..\house-cognition-data\set\';
files = dir(fullfile(path, '*.set'));

filenames = {files.name}';
len_names = size(filenames, 1);

for k = 1: len_names
    trace = strcat(path, filenames(k));
    [filepath, name, ext] = fileparts(trace);
    EEG = pop_loadset(trace{1,1});
    EEG = pop_eegfiltnew(EEG, 0, 40, [], 0, 0, 0);
    EEG = pop_resample(EEG, 250);
    data = EEG.data;
    %%4 average reference
    data = data - repmat(mean(data),size(data,1),1);
    tensor = data(:, 26:425, :);  % [-100, 1500]
    disp(name)
    file_mat = char(name(1,:));
    trials = EEG.trials;
    epoch = EEG.epoch;
    rt = [];
    try
        for i=1:trials
            eventtype = epoch(i).eventtype;
            rt_type = find(strcmp(eventtype, 'S254'));
            if isempty(rt_type)
                rt_type = find(strcmp(eventtype, 'S251'));
            end
            eventlatency = epoch(i).eventlatency;
            reaction_time = eventlatency(rt_type);      % extract latency
            rt = [rt, reaction_time{1,1}];
        end

        save(['..\..\..\house-cognition-data\mat\', file_mat, '.mat'], 'tensor', 'rt');
    catch
        disp('something wrong')
    end
end
