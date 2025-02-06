function results = RIDE_call_modified(data,cfg)

for copyright = 1:1
% Code Author: Guang Ouyang, HKBU, 2010,
% The RIDE method developers: Guang Ouyang, Werner Sommer, Changsong Zhou (alphabetical order)
% Copyright (C) <2010>  Guang Ouyang, Werner Sommer, Changsong Zhou
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
end

%-------------save the initial configurations------------------------------
cfg_raw = cfg; 
if length(cfg.comp.name)==1 
    results.erp = mean(data,3);return;%if only separate one component, it is ERP
end

%-------------preparation, down_sampling-----------------------------------
for section = 1:1  
    [d1,d2,d3] = size(data);    %[tdim, channels, trials]
    epoch_length = d1;
    erp = mean(data,3);
    results.erp = erp;          % original ERP
    results.latency0 = cfg.comp.latency;%save the original latency information, i.e., RT

    %------------down sample data------------------------------------------
    rs = (cfg.re_samp/cfg.samp_interval);
    data = data(round(linspace(1,d1,fix(d1/rs))),:,:);
    
    %----------only if using specified template to measure C---------------
    if isfield(cfg,'template')
        if strcmp(cfg.template.method,'g_mean')
            load(cfg.temp_path);
            template = template(round(linspace(1,d1,fix(d1/rs))),:,:);
        end
    end
    %----------------------------------------------------------------------

    [d1,d2,d3] = size(data);%new size after down samping
    
    
    %----------only for microsaccade---------------------------------------
    if isfield(cfg,'latency_a') 
        cfg.latency_a = round(cfg.latency_a/cfg.re_samp);
        cfg.latency_a = round(cfg.latency_a-median(cfg.latency_a(~isnan(cfg.latency_a))));
        cfg.ms_twd = fix((cfg.ms_twd-cfg.epoch_twd(1))/cfg.re_samp)+[1,-1];
    end

    %--------unified the unit of time window and latency information-------
    for j = 1:cfg.comp_num
        if cfg.comp.latency{j} == 0 
            cfg.comp.latency{j} = zeros(d3,1);
        end
        if ~ischar(cfg.comp.latency{j})
            if strcmp(cfg.comp.name{j},'r')
                cfg.comp.twd{j} = cfg.comp.twd{j} + median(results.latency0{j});
                cfg.comp.twd{j}(cfg.comp.twd{j}<cfg.rwd) = cfg.rwd; %left boundary of twd for R not less then 200 ms
                cfg.comp.twd{j}(cfg.comp.twd{j}>cfg.epoch_twd(2)) = cfg.epoch_twd(2);
            end
            cfg.comp.latency{j} = ((cfg.comp.latency{j})/cfg.re_samp);
            cfg.comp.latency{j} = round(cfg.comp.latency{j} - median(cfg.comp.latency{j}));
        end 
        cfg.comp.twd{j} = fix((cfg.comp.twd{j}-cfg.epoch_twd(1))/cfg.re_samp)+[1,-1];%convert component time window to sampling unit
        if cfg.comp.twd{j}(2)<cfg.comp.twd{j}(1) 
            cfg.comp.twd{j} = [cfg.comp.twd{j}(2)-1 cfg.comp.twd{j}(2)]';
        end
    end
    
    %--------specify the searching duration--------------------------------
    if isfield(cfg,'dur')
        for j = 1:length(cfg.comp.name)
            cfg.dur{j} = fix(cfg.dur{j}/cfg.re_samp);
        end
    else
        for j = 1:length(cfg.comp.name)
           cfg.dur{j} = round(((cfg.comp.twd{j}(2)-cfg.comp.twd{j}(1)))/2);
        end
    end
end
    
        

%---------initial estimation of the latency of C---------------------------   
for section = 1:1
    for initial_c = 1:1 
        n_of_c = 0;
        c_i = 0;
        for j = 1:cfg.comp_num
            if ischar(cfg.comp.latency{j})
                if cfg.prg == 1 
                    disp(['woody_for_',cfg.comp.name{j}]);
                end
                n_of_c = n_of_c + 1;
                c_i(n_of_c) = j;
                final_correlations = 1:d2;
                current_template = cfg.comp.twd{j};
                if isfield(cfg,'template')
                    if strcmp(cfg.template.method,'g_mean')
                        cfg.temp = template(current_template(1):current_template(2),final_correlations);
                    end
                    if isfield(cfg.template,'chan')
                        final_correlations = cfg.template.chan;
                        if isfield(cfg.template,'hann_amp')
                            cfg.template.hann_amp = cfg.template.hann_amp(cfg.template.chan);
                        end
                    end
                end
                %-------using Woody's method by default
                cfg.comp.latency{j} = woody(data(current_template(1):current_template(2),final_correlations,:),cfg,cfg.dur{j});
            end
        end
    end
end

    
%------------RIDE iteration------------------------------------------------
for j=1:cfg.comp_num
    cfg.comp.latency{j} = repmat(cfg.comp.latency{j}, 1, 41);
    cfg.comp.latency{j} = cfg.comp.latency{j}.';
end
for section = 1:1 
    stop=0;
    
    for j = 1:n_of_c% track latency evolution of C components
        latency_i{j}(:,:) = cfg.comp.latency{c_i(j)}(:,:);
        % latency_i{j}(:,:) = latency_i{j}(:);
        l_change(:,:,j) = ones(d2, d3);%track for evolution of the latency in order to terminate the iteration
        c_change(:,:,j) = ones(d2, d3);%track for evolution of the correlation in order to terminate the iteration
    end

    if cfg.prg == 1 
        disp('RIDE decomposition: ');
    end

    outer_iter = 4;
    if n_of_c == 0 %outer iteration is empericaly limited to 4, but if no c component, no need to do outer iteration
        outer_iter = 1;
    end


    for iter = 1:outer_iter
        
         %report the progress
         if n_of_c ~= 0 
             prog = fix(100*(1-sum(mean(l_change,3))/d3*(10-iter)/10));
             if cfg.prg == 1 
                 %barh(prog);axis([0,100,0.5,1.5]);axis off;text(10,1.5,strcat('iteration','--',num2str(prog),'%done'));pause(0.001);
                 fprintf(strcat('iteration',num2str(iter),'--',num2str(prog),'%%done\n'));
             end
             if (fix(100*(1-sum(mean(l_change,3))/d3*(10-iter)/10)))>=99
                 stop=1;%stop the iteration when more 99% of the single trial latency do not change
             end
         end

         if cfg.prg == 1 
             fprintf('iteration step for each channel:\n');
         end
         for j = 1:n_of_c        
             latency_i{j}(:,:,iter) = cfg.comp.latency{c_i(j)};         
         end
         if iter == outer_iter 
             stop=1;
         end

         
         %---------------start to RIDE-------------------------------------
         for sec_RIDE_inner_iter = 1:1
             cfg1 = cfg;
             cfg1.final = stop;
             cfg1.inner_iter = 100;%this is to safeguard the extreme
             %iterations (usually inner iteration stops before 20)
             
             c_l = zeros(d1, d2, cfg.comp_num); %c_l is the latency-locked RIDE component
             c_sl = c_l; %c_sl is the stimulus-locked RIDE component
             for channel = 1:d2
                 
                 rst = RIDE_iter(squeeze(data(:,channel,:)), cfg1, channel); %RIDE decomposition
                 
                 c_l(:,channel,:) = rst.comp;
                 c_sl(:,channel,:) = rst.comp1;
                 
                 if stop==1  
                     amp(:,channel,:) = permute(rst.amp,[1,3,2]);
                 end
                 
                 if isfield(cfg,'latency_a')%only for microsaccade 
                     comp_ms(:,channel) = rst.comp_ms;
                     comp_ms1(:,channel) = rst.comp_ms1;
                 end               
             end
             comp = c_l;
             comp1 = c_sl;
         end
         %-----------------------------------------------------------------
         
         
        
         %--------------latency estimation of C----------------------------
         for section1 = 1:1
            for c_idx = 1:n_of_c
                clear channel_correlations;
                no_p{c_idx} = ones(d3,1);%check there are peaks in x-corr curves or not
                %low pass filter the template only for
                %calculation of cross-correlation curve
                current_template = filtering20(comp(:,:,c_i(c_idx)),1,round(10*cfg.high_cutoff*d1*cfg.re_samp/1000));
                
                channels = 1:d2; 
                for channel = 1:length(channels) %cross correlation
                    for trial = 1:d3 %trial_index
                        if l_change(channel, trial, c_idx)==1
                            single_trial = data(:,channel,trial);
                            for j = 1:cfg.comp_num
                                if j~=c_i(c_idx)
                                    single_trial = single_trial - move2(comp(:,:,j),cfg.comp.latency{j}(channel, trial),'1d');
                                end
                            end
                            if isfield(cfg,'latency_a')%remove the ms component
                                for a = 1:length(find(~isnan(cfg.latency_a(:,trial))))
                                        single_trial = single_trial - move2(comp_ms,cfg.latency_a(a,trial),'1d'); 
                                end
                            end
                            
                            %low pass filter the data only for calculation of cross-correlation curve
                            single_trial = filtering20(single_trial,1,round(10*cfg.high_cutoff*d1*cfg.re_samp/1000));
    
                            %remove the linear trend to safeguard that the 
                            %latency estimation is not affected by drifting
                            single_trial = RIDE_detrend(single_trial,[cfg.comp.twd{c_i(c_idx)}(1),cfg.comp.twd{c_i(c_idx)}(1)+fix((cfg.comp.twd{c_i(c_idx)}(2)-cfg.comp.twd{c_i(c_idx)}(1))*cfg.bd),...
                                cfg.comp.twd{c_i(c_idx)}(1) + fix((cfg.comp.twd{c_i(c_idx)}(2)-cfg.comp.twd{c_i(c_idx)}(1))*(1-cfg.bd)), cfg.comp.twd{c_i(c_idx)}(2)]);
    
                            channel_correlations{c_idx}(:,trial) = xcov(single_trial(:),current_template(:,channels(channel)),fix(size(single_trial,1)/2),cfg.xc);
        
                            %low pass filter the cross correlation curve
                            channel_correlations{c_idx}(:,trial) = filtering10(channel_correlations{c_idx}(:,trial),1,round(10*cfg.high_cutoff*size(single_trial,1)*cfg.re_samp/1000));
                        end
                    end

                    %detrend the cross-correlation curve to make sure the latency will not be found on the boundaries
                    channel_correlations{c_idx} = RIDE_detrend(channel_correlations{c_idx}(:,:), [1, fix(size(channel_correlations{c_idx},1)*cfg.bd),fix(size(channel_correlations{c_idx},1)*(1-cfg.bd)),size(channel_correlations{c_idx},1)]);
                    final_correlations = [];
                    correlation_interval = channel_correlations{c_idx}(fix(size(channel_correlations{c_idx},1)/2-cfg.dur{c_i(c_idx)})+1:fix(size(channel_correlations{c_idx},1)/2+cfg.dur{c_i(c_idx)}),:);
                    
                    if strcmpi(cfg.latency_search,'most_prob') %search the nearest peak from the most probable estimation
                        for j = 1:d3 
                            final_correlations(j) = nearest_latency(correlation_interval(:,j),cfg.comp.latency{c_i(c_idx)}(j)+fix(size(correlation_interval,1)/2));
                        end
                    end
                    if strcmpi(cfg.latency_search,'all')%search the largest peak
                        for j = 1:d3 
                            final_correlations(j) = find_peak(correlation_interval(:,j));
                        end
                    end
    
                    
                    % randomly assign the latency of the trials without x-corr peak
                    for j = 1:d3 
                        if no_peak(correlation_interval(:,j))==0 
                            no_p{c_idx}(j) = 0;
                        end
                    end
    
                    %make sure not exceed boundary      
                    final_correlations(no_p{c_idx}==0) = round(randn(length(find(no_p{c_idx}==0)),1)*std(final_correlations(no_p{c_idx}==1))) + fix(size(correlation_interval,1)/2);
                    final_correlations(final_correlations<1)=1;
                    final_correlations(final_correlations>size(correlation_interval,1)) = size(correlation_interval,1);
    
                    %track the correlation values
                    for j = 1:d3 
                        corr_i{c_idx}(j,iter) = correlation_interval(final_correlations(j),j);
                    end 
                   
                    %covert the C latencies to relative values by subtracting the median
                    final_correlations = round(final_correlations-median(final_correlations)); 
    
    
                    %track the latency evolution and correlation values (if the evolution returns, then stop updating)   
                    if iter>1
                        for j = 1:d3
                            if (final_correlations(j)-latency_i{c_idx}(j,iter))*(latency_i{c_idx}(j,iter)-latency_i{c_idx}(j,iter-1))<=0&&l_change(j,c_idx)==1 
                                l_change(channel, j, c_idx) = 0;
                            end
                            c_change(channel, j,c_idx)=1;
                            for jj = 1:iter-1 
                                if corr_i{c_idx}(j,jj)>=corr_i{c_idx}(j,iter)    
                                    c_change(channel, j,c_idx)=0;        
                                end
                            end
                        end
                     end
    
            
                     index = (l_change(channel,:,c_idx)==1&c_change(channel,:,c_idx)==1);
                     cfg.comp.latency{c_i(c_idx)}(channel, index) = final_correlations(index);
                     cfg.comp.latency{c_i(c_idx)} = round(cfg.comp.latency{c_i(c_idx)}(channel,:)-median(cfg.comp.latency{c_i(c_idx)}(channel,:)));
                     
                     disp('d')
                  
                end
            end
         end

         if stop==1
             break;
         end
    end
    if cfg.prg == 1 
        fprintf('100%%done\n');
        close(gcf);
    end
end



%---------------final data-------------------------------------------------
for section = 1:1

    %if data has been down sampled, apply interpolation to restore the
    %original resolution and re-baselining
    results.erp_new = 0;
    results.residue = erp;
    if isfield(cfg,'latency_a') %only for microsaccades
        results.ms = baseline(interp2d(comp_ms,round(linspace(1,epoch_length,d1)),1:epoch_length,'spline'));
        results.ms_sl = baseline(interp2d(comp_ms1,round(linspace(1,epoch_length,d1)),1:epoch_length,'spline'));
    end

    bl_wd = fix(-cfg.epoch_twd(1)/cfg.samp_interval)+1:fix(-cfg.epoch_twd(1)/cfg.samp_interval+cfg.bl/cfg.samp_interval);%baseline time window
    for j = 1:cfg.comp_num
        component(:,:,j) = interp2d(comp(:,:,j),round(linspace(1,epoch_length,d1)),1:epoch_length,'spline');
        component(:,:,j) = baseline(component(:,:,j),bl_wd);
        component1(:,:,j) = interp2d(comp1(:,:,j),round(linspace(1,epoch_length,d1)),1:epoch_length,'spline');
        component1(:,:,j) = baseline(component1(:,:,j),bl_wd);
        results.residue = results.residue - component1(:,:,j);
        eval(['results.',cfg.comp.name{j},' = component(:,:,j);']);
        eval(['results.',cfg.comp.name{j},'_sl = component1(:,:,j);']);
        eval(['results.latency_',cfg.comp.name{j},' = cfg.comp.latency{j}*cfg.re_samp;']);
        eval(['results.amp_',cfg.comp.name{j},'=amp(:,:,j);']);
    end
    
    if isfield(cfg,'latency_a') %only for microsaccades
        results.residue = results.residue - results.ms_sl;
    end
    
    eval(['results.',cfg.comp.name{1},' = baseline(results.',...
        cfg.comp.name{1},' ,bl_wd) + repmat(mean(erp(bl_wd,:)),[epoch_length,1]);']);
    eval(['results.',cfg.comp.name{1},'_sl = baseline(results.',...
        cfg.comp.name{1},'_sl,bl_wd) + repmat(mean(erp(bl_wd,:)),[epoch_length,1]);']);  
    eval(['results.',cfg.comp.name{rst.trend_c},' = baseline(results.',...
        cfg.comp.name{rst.trend_c},' + results.residue,bl_wd);']);
    eval(['results.',cfg.comp.name{rst.trend_c},'_sl = baseline(results.',...
        cfg.comp.name{rst.trend_c},'_sl + results.residue,bl_wd);']);
    
    if cfg.comp_num == 1 
        bl_wd = 1:-fix(cfg.epoch_twd(1)/cfg.samp_interval);
        eval(['results.',cfg.comp.name{1},' = baseline(results.',...
        cfg.comp.name{1},' + results.residue,bl_wd) + repmat(mean(erp(bl_wd,:)),[epoch_length,1]);']);
        eval(['results.',cfg.comp.name{1},'_sl = baseline(results.',...
            cfg.comp.name{1},'_sl + results.residue,bl_wd) + repmat(mean(erp(bl_wd,:)),[epoch_length,1]);']);
    end
    for j = 1:cfg.comp_num 
        eval(['results.erp_new = results.erp_new + results.',cfg.comp.name{j},';']);
    end
    if n_of_c~=0 
        results.latency_i = latency_i;
        results.no_p = no_p;
    end
    results.cfg = cfg_raw;
    if exist('corr_i','var') 
        results.corr_i = corr_i;
    end
end


