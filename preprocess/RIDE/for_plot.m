%% plot sorted unsynchronized and synchronized C components within subjects
%let compare pre_sync and post_sync C data (sorted by C latency):
stage12_path   = '../../../face-cognition-data/step12/';
files = dir(fullfile(stage12_path, '*.mat'));
filenames = {files.name}';
len_names = size(filenames, 1);
load("./channel.mat");

for k = 1: len_names
    trace = strcat(stage12_path, filenames(k));
    [filepath, name, ext] = fileparts(trace);
    eval(['load(trace{1,1})', ';']);
    
    ch_label = 'Pz';
    ch_index = find(strcmpi({chanlocs.labels}, ch_label));
    [tem1,tem2] = sort(results.latency_c_idx);
    
    f = figure('visible','off');
    subplot(1,2,1);
    data=results.st_c;
    imagesc(t_axis,1:size(data,3),squeeze(data(:,ch_index,tem2))');colormap(jet);
    xlabel('time after stimulus (ms)');
    ylabel('trial (sorted)');
    title('single trial C');
    
    subplot(1,2,2);
    data=results.st_c_synced;
    imagesc(t_axis,1:size(data,3),squeeze(data(:,ch_index,tem2))');colormap(jet);
    xlabel('time after stimulus (ms)');
    ylabel('trial (sorted)');
    title('single trial C (synced)');

    save_fig_name = '../../../syn/' + convertCharsToStrings(name);
    saveas(f, save_fig_name, 'jpg')
    
end


%%
%plot single trial C
chan_index = find(strcmpi({chanlocs.labels},'Pz'));
disp(accending_index(results.amp_c(:,chan_index)))
data = results.st_c;   
t_axis = linspace(-100, 1500, size(data,1));%time axis;
% temp = single_trial_RIDE(data,results,'c',chan_index);
temp = data(:,chan_index,:);
%sort by the accending order of C latency

figure;imagesc(t_axis,1:size(data,3),temp(:,accending_index(results.latency_c))');
xlabel('time after stimulus (ms)');ylabel('trial index');title('accending order of latency - C');

%sort by the accending order of amplitude
figure;imagesc(t_axis,1:size(data,3),temp(:,accending_index(results.amp_c(:,chan_index)))');
xlabel('time after stimulus (ms)');ylabel('trial index');title('accending order by amplitude - C');
disp(accending_index(results.amp_c(:,chan_index)));




