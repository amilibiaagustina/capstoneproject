%% ERP analysis
% ERP  Analysis for initial periods of each ERP trial (Exposure, Compulsion,
% Relief, etc.)

clear all;
clc;

%% Load in .mat files from ERP sessions
[filename,location] = uigetfile({'*.mat'},'Select an .mat file', '/Volumes/datalake/AA-56119/AA002/clinic/2025-04-10/preprocessed/ERP'); %DBSPsych-56119/DBSOCD002/clinic/2025-04-10/preprocessed/ERP/PDBSOCD002_2025-04-10_ERP_1.mat';
load(fullfile(location, filename))

% Subject ID
split_filename = split(filename, '/');
sub_name_date = extractBefore(split_filename{end,1}, '.mat');

%open video
videoFile = '/Users/agusamilibia/Library/CloudStorage/GoogleDrive-ma223@rice.edu/My Drive/BCM/Matlab/camera-sync-master-BV2/videos/DBSOCD001_2025-02-14_camera11_1.mp4';
videoObj = VideoReader(videoFile);

%% Set output directory
current_dir = pwd;
output_folder = fullfile(current_dir, sub_name_date);

if not(isfolder(output_folder))
    mkdir(output_folder)
end

%% extract LFP data 
% Find the column index for 'TD_Aic' in the combined data table
LFP_table_column = find(contains(data.neural.combined_data_table.Properties.VariableNames, 'TD_Aic'));

% Check if 'ONE' exists in the variable names and assign LFP data accordingly
if(find(contains(data.neural.combined_data_table.Properties.VariableNames, 'ONE')))
    LFP_Left = data.neural.combined_data_table.TD_Aic_ONE_THREE_LEFT;
    LFP_Right = data.neural.combined_data_table.TD_Aic_ONE_THREE_RIGHT;
else
    LFP_Left = data.neural.combined_data_table.TD_Aic_ZERO_TWO_LEFT;
    LFP_Right = data.neural.combined_data_table.TD_Aic_ZERO_TWO_RIGHT;
end

% Demean the LFP data by subtracting the mean, ignoring NaN values
LFP_demeaned_Left = LFP_Left - mean(LFP_Left, 'omitnan');
LFP_demeaned_Right = LFP_Right - mean(LFP_Right, 'omitnan');

% Split the demeaned LFP data and retrieve additional information
[LFP_demeaned_split_Left, LFP_Nans_Left, total_chunks_Left, beginning_nans, end_nans] = LFP_demeaned_split(LFP_demeaned_Left);
[LFP_demeaned_split_Right, LFP_Nans_Right, total_chunks_Right, beginning_nans, end_nans] = LFP_demeaned_split(LFP_demeaned_Right);

%% timestamps
timestamps = ((1/data.neural.fs): (1/data.neural.fs): (length(LFP_demeaned_Left)/data.neural.fs));

%% set constants
window = round(.5*data.neural.fs);
noverlap = round(.75*window);
nfft = max(256, 2^nextpow2(window)); % Power of 2 is faster

%% plot all timechunks
for num_chunk = 1:total_chunks_Left
    % spectrogram
    % Compute the spectrogram
    [S_Left, F_Left, T_Left] = spectrogram(LFP_demeaned_split_Left{1,num_chunk}, hamming(window), noverlap, nfft, data.neural.fs, 'yaxis');
    [S_Right, F_Right, T_Right] = spectrogram(LFP_demeaned_split_Right{1,num_chunk}, hamming(window), noverlap, nfft, data.neural.fs, 'yaxis');

    % Compute the log scale of the spectrogram (dB scale)
    log_S_Left = abs(10*log10(abs(S_Left))); 
    log_S_Right = abs(10*log10(abs(S_Right))); 

    % get current timestamps
    if(~isempty(LFP_Nans_Left))
        if num_chunk == 1
            curr_timestamps = timestamps(1:(beginning_nans(num_chunk))-1);
        elseif (num_chunk == total_chunks_Left)
            curr_timestamps = timestamps((end_nans(num_chunk-1)+1):length(LFP_demeaned_Left));
        else
            curr_timestamps = timestamps((end_nans(num_chunk-1)+1):(beginning_nans(num_chunk))-1);
        end
    else
        curr_timestamps = timestamps;
    end

    % make figure
    timeseries_fig = figure;
    tiledlayout(3,2)
    p1 = nexttile;
    plot(curr_timestamps,LFP_demeaned_split_Left{1,num_chunk})
    title(strcat('Chunk: ', num2str(num_chunk)), ' Left Time Frequency Plot')
    ylim([-30,30])
    p4 = nexttile;
    plot(curr_timestamps,LFP_demeaned_split_Right{1,num_chunk})
    title(strcat('Chunk: ', num2str(num_chunk)), ' Right Time Frequency Plot')
    ylim([-30,30])
    p2 = nexttile;
    imagesc(curr_timestamps, F_Left, log_S_Left);
    title(strcat('Chunk: ', num2str(num_chunk)), ' Left Spectrogram')
    colorbar;
    p5 = nexttile;
    imagesc(curr_timestamps, F_Right, log_S_Right);
    title(strcat('Chunk: ', num2str(num_chunk)), ' Right Spectrogram')
    colorbar; 
    p3 = nexttile;
    plot(curr_timestamps,bandpass(LFP_demeaned_split_Left{1,num_chunk},[4,8], data.neural.fs))
    title(strcat('Chunk: ', num2str(num_chunk)), ' Left Theta Bandpass') 
    p6 = nexttile;
    plot(curr_timestamps,bandpass(LFP_demeaned_split_Right{1,num_chunk},[4,8], data.neural.fs))
    title(strcat('Chunk: ', num2str(num_chunk)), ' Right Theta Bandpass') 
    linkaxes([p1,p2,p3,p4,p5,p6], 'x')
    linkaxes([p2,p5], 'xy')


    savefig(timeseries_fig, fullfile(output_folder, strcat(sub_name_date, '_Chunk_', num2str(num_chunk))))

end