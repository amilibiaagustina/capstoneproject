%% ERP analysis
% ERP  Analysis for initial periods of each ERP trial (Exposure, Compulsion,
% Relief, etc.)

clear;
clc;

%% Load in .mat files from ERP sessions
[filename,location] = uigetfile({'*.mat'},'Select an .mat file', '/Volumes/datalake/DBSPsych-56119/DBSOCD001/clinic/2025-02-14/preprocessed/ERP'); %DBSPsych-56119/DBSOCD002/clinic/2025-04-10/preprocessed/ERP/PDBSOCD002_2025-04-10_ERP_1.mat';
load(fullfile(location,filename));

% Subject ID
split_filename = split(filename, '/');
sub_name_date = extractBefore(split_filename{end,1}, '.mat');

%open video
videoFile = '/Users/agusamilibia/Documents/MATLAB/camera-sync-master-BV2/videos/DBSOCD001_2025-02-14_camera11_1.mp4';
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

%Replace Nan by 0
LFP_demeaned_Left(isnan(LFP_demeaned_Left)) = 0;
LFP_demeaned_Right(isnan(LFP_demeaned_Right)) = 0;

%% timestamps
curr_timestamps = ((1/data.neural.fs): (1/data.neural.fs): (length(LFP_demeaned_Left)/data.neural.fs));

%% set constants
window = round(.5*data.neural.fs);
noverlap = round(.75*window);
nfft = max(256, 2^nextpow2(window)); % Power of 2 is faster

x= curr_timestamps;
LFP_r = LFP_demeaned_Right;
LFP_l = LFP_demeaned_Left;
theta_L = bandpass(LFP_demeaned_Left, [4 8], data.neural.fs);
theta_R = bandpass(LFP_demeaned_Right, [4 8], data.neural.fs);

%% plot LFPs

timeseries_fig = figure;
axis([-5,5,-30,30]);
h1 = animatedline;
grid on;
drawnow;

for k = 1:length(x)
    % Update the animated lines with current data
    addpoints(h1, x(k), LFP_r(k));
      if k >1250
        xlim([-5+x(k),5+x(k)])      
      end
      pause(0.0001)

    drawnow limitrate

end

savefig(timeseries_fig, fullfile(output_folder, strcat(sub_name_date, '_DynamicFullPlot')));


    %addpoints(h2, x(k), LFP_l(k));
    %addpoints(h3, x(k), theta_R);
    %addpoints(h4, x(k), theta_L);

    % Update the plot limits dynamically
    
    % Optionally, pause for a brief moment to visualize the updates
    %pause(0.01);
