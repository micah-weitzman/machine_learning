function [train_set, train_label, test_set, test_label] = machine_learning(data, seizure_layer, percent_train, t2s, window_length, overlap, time_between_seizures, time_started, hours_from_gmt, inter, username, pswd)


%MACHINE_LEARNING Annalyzes EEG data from seizures using machine learning to predict seizures
%
% Syntax:  [train, inter_data] = machine_learning(data, seizure_layer, 
%     percent_train, t2s, window_length, overlap, time_between_seizure, 
%     time_started, hours_from_gmt, inter, username, pswd)
%
% Inputs:
%    data - IEEGDataset for analysis
%    seizure_layer - IEEG annotation layer to select from 
%    percent_train - Percentage of data selected to train the machine
%    t2s (hour) - Time to seizure
%    window_length (seconds) - Size of window frames
%    overlap (seconds) - Overlap between windows
%    time_between_seizures (day) - Time between seizure
%    time_started (24 hour) - Start of recording in Unix Time
%    hours_from_gmt (+ or - in hours) - Difference in timezone from GMT
%    inter (percent) - Percent of interictal segments to analyize 
%    username - IEEG username to connect to database
%    pswd - IEEG password to connect to database
%
% Outputs:
%    train_set - training dataset 
%    train_label - True or false labels for pre-ictal training
%    test_set - testing dataset
%    test_label - True or false labels for pre-ictal testing
%
% Example: 
%    Line 1 of example
%    Line 2 of example
%    Line 3 of example
%

% Author: Micah Weitzman
% email: mikachoow21@gmail.com
% Website: http://www.micahweitzman.com
% August 2016; Last revision: August-2016

%------------- BEGIN CODE --------------

    % Connect to IEEG database
    global session;
    session =  IEEGSession(data, username, pswd);

    % Get all sizure start times and stop times
    seizure_start_times = {session.data(1).annLayer(seizure_layer).getEvents(0).start};
            % Strange syntax hack
    stop_times = {session.data(1).annLayer(seizure_layer).getEvents(0).stop};
    seizure_stop_times = stop_times(1:ceil(size(stop_times, 2) * (inter * percent_train)/10000));
    % Retrieve sample rate 
    fs = session.data(1).sampleRate;
    
    % Retrieve number of sizures to train on provided percentage to train
    seizures_to_train = seizure_start_times(1:ceil(size(seizure_start_times, 2) * percent_train/100));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%% TRAINING %%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % TRAIN PRE
    % Matrix of data used for training 
    train = [];
    for seizure = 1:length(seizures_to_train) 
        s_time = seizures_to_train{1,seizure}(1)- t2s * 3600 * 1000000;
        
        val = session.data(1).getvalues(double(s_time), t2s * 3600 * 1000000, 1:16);
        train(:,:,seizure) = analyze_data(val, 'training', size(seizures_to_train, 2));
    end

    hist_of_times = reshape(train(:,3,:), [], 1);
    % set clearnece between post-seizure and pre-seizure
    cl = 8.64e+10 * time_between_seizures;

    max_time = seizures_to_train{end}(1);
    
    % TRAIN INTER 
    inter_data = [];
    % find segments of data during interictal phase to anaylize 
    num_inter_clips = inter/100 * length(seizures_to_train);
    for seizure = 1:num_inter_clips
        p=0;
        s_time=0;
        while p ~= 1
            s_time = randi(max_time, 1);
            for foo = 1:size(seizures_to_train)
                if  (s_time > seizures_to_train{foo}(1)) && (s_time < stop_times{foo}(1))
                elseif (s_time > stop_times{foo}(1) + cl) && (s_time < seizures_to_train{foo + 1}(1)-cl)
                    p = 1;
                end
            end
        end
        
        X = randi(length(hist_of_times));
        X = hist_of_times(X);
        
        T = datestr((s_time / 1000000 + time_started + hours_from_gmt * 3600)/86400 + datenum(1970,1,1));
        T = str2num(T(13:14));
        if double(T) < double(X)
            s_time = s_time + (X-T)*3600000000;
        else
            s_time = s_time - (T-X)*3600000000;
        end
        
        val = session.data(1).getvalues(double(s_time), t2s * 3600 * 1000000, 1:16);
        inter_data(:,:,seizure) = analyze_data(val, 'training', length(seizure_stop_times));
    end
    
    % Training Data output. The dataset and the labels
    train_1 = reshape(permute(train, [1 3 2]),size(train,1)*size(train,3),size(train,2));
    train_2 = reshape(permute(inter_train, [1 3 2]),size(inter_train,1)*size(inter_train,3),size(inter_data,2));
    train_set1 = train_1(~any(isnan(train_1),2),:);
    train_set2 = train_2(~any(isnan(train_2),2),:);
    train_set = [train_set1 ; train_set2];
    train_label(1:size(train_1)) = 1;
    train_label(end:end+size(train_1)) = 0;
    train_label = train_label';
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%% TESTING %%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    seizures_to_test = setdiff([seizure_start_times{:}], [seizures_to_train{:}]);
    test_stop_times = [stop_times(length(seizures_to_train):end)];
    
    % TEST PRE
    test = [];
    for seizure = 1:length(seizures_to_test) 
        s_time = seizures_to_test(seizure)- t2s * 3600 * 1000000;
             
        val = session.data(1).getvalues(s_time, t2s * 3600 * 1000000, 1:16);
        test(:,:,seizure) = analyze_data(val, 'testing',  size(seizures_to_test, 2));
    end
    
    hist_of_times = reshape(test(:,3,:), [], 1);
    % set clearnece between post-seizure and pre-seizure
    cl = 8.64e+10;

    % TEST INTER
    inter_test = [];
    % find segments of data during interictal phase to anaylize 
    for seizure = 1:num_inter_clips
        p=0;
        s_time=0;
        while p ~= 1
            s_time = randi([test_stop_times{1}(1) seizures_to_test(end)], 1);
            for foo = 1:size(seizures_to_test)
                if  (s_time > seizures_to_test(foo)) && (s_time < test_stop_times{foo}(1))
                    p = 0;
                elseif (s_time > test_stop_times{foo}(1) + cl) && (s_time < seizures_to_test(foo + 1)-cl)
                    p = 1;
                end
            end
        end
        
        T = datestr((s_time / 1000000 + time_started + hours_from_gmt * 3600)/86400 + datenum(1970,1,1));
        T = str2num(T(13:14));
        if double(T) < double(X)
            s_time = s_time + (X-T)*3600000000;
        else
            s_time = s_time - (T-X)*3600000000;
        end
        
        
        val = session.data(1).getvalues(s_time, t2s * 3600 * 1000000, 1:16);
        inter_test(:,:,seizure) = analyze_data(val, 'testing',  size(seizure_stop_times, 2));
    end
    
    % Training Data output. The dataset and the labels\
    test_1 = reshape(permute(test, [1 3 2]),size(test,1)*size(test,3),size(test,2));
    test_2 = reshape(permute(inter_test, [1 3 2]),size(inter_test,1)*size(inter_test,3),size(inter_data,2));
    test_set1 = test_1(~any(isnan(test_1),2),:);
    test_set2 = test_2(~any(isnan(test_2),2),:);
    test_set = [test_set1 ; test_set2];
    test_label(1:size(test_1)) = 1;
    test_label(end:end+size(test_1)) = 0;
    test_label = test_label';

    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    %%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    
    % Function that takes values, dataset, testing or training (string
    % used for printing to console), and number of seizures and does common
    % analysis 
    function [dataset] = analyze_data(values, test_or_train, num_of_seizure)
        time = 1;
        i = 1;
        while time < size(values, 1) - window_length*fs
            windo = values(time:round(time+window_length*fs), :);

            t = datestr(((s_time + (time/fs)*1000000) / 1000000 + time_started + hours_from_gmt * 3600)/86400 + datenum(1970,1,1));
            hours_time = str2num(t(13:14));
            minutes_time = str2double(t(16:17));
            if minutes_time > 30
                hours_time = hours_time + 0.5;
            end 

            % find mean and standard distribution for each frame
            m = mean(nanmean(windo));
            sd = std(nanstd(windo));
            
            
            windo = windo(~any(isnan(windo),2),:);
            if isempty(windo) == 0
                co = corr(windo);
                tri = reshape(triu(co, 1), 1, []);
                tri = tri(tri~=0);
            else
                tri(1, 1:50) =  NaN;
            end

            dataset(i,:) = [m sd hours_time, tri(1:50)];

            fprintf('[%s] Finished %d of seizure %d of %d\n', test_or_train, i, seizure, num_of_seizure);

            i = i + 1;
            time = time + window_length * fs - overlap * fs;
        end 
    end
end