%% ERP analysis
% ERP  Analysis for initial periods of each ERP trial (Exposure, Compulsion,
% Relief, etc.)

clear; 
clc;
tic;
% add path to custom FFmpeg functions (used later for video/audio processing)
ffmpeg_path = '/opt/homebrew/bin/ffmpeg';


%% load in .mat files from ERP sessions
[filename,location] = uigetfile({'*.mat'},'Select an .mat file', '/Users/agusamilibia/Library/CloudStorage/Box-Box/datalake/AA-56119/AA001/clinic/2025-01-28/preprocessed/ERP');
load(fullfile(location,filename));
% subject ID
split_filename = split(filename, '/');  
sub_name_date = extractBefore(split_filename{end,1}, '.mat');
sub_name_date = strrep(sub_name_date, 'PDBSOCD', 'AA');

%% extract LFP data 
% find the column index for 'TD_Aic' in the combined data table
LFP_table_column = find(contains(data.neural.combined_data_table.Properties.VariableNames, 'TD_Aic'));

%neural alignment with video (beep is on event 10)
target_ts = data.behavior.events.Timestamp(data.behavior.events.Event_Code == 10); %10 is start of task (video beep)
ts = data.neural.combined_data_table.Timestamp;%all timestamps from the neural combined data table
[~, idx_start] = min(abs(ts - target_ts)); %find the index of the timestamp closest to the target
%Get the corresponding datetime for reference
datetime_value = data.neural.combined_data_table.Datetime(idx_start);


% check for LFP variable names and assign accordingly
varNames = data.neural.combined_data_table.Properties.VariableNames;

% try different naming patterns 
if any(contains(varNames, 'TD_Aic_ONE_THREE_LEFT'))
    LFP_Left = data.neural.combined_data_table.TD_Aic_ONE_THREE_LEFT(idx_start:end);
    LFP_Right = data.neural.combined_data_table.TD_Aic_ONE_THREE_RIGHT(idx_start:end);
    fprintf('Using ONE_THREE channels\n');
elseif any(contains(varNames, 'TD_Aic_ZERO_TWO_LEFT'))
    LFP_Left = data.neural.combined_data_table.TD_Aic_ZERO_TWO_LEFT(idx_start:end);
    LFP_Right = data.neural.combined_data_table.TD_Aic_ZERO_TWO_RIGHT(idx_start:end);
    fprintf('Using ZERO_TWO channels\n');
elseif any(contains(varNames, 'TD_Other_ZERO_TWO_LEFT'))
    LFP_Left = data.neural.combined_data_table.TD_Other_ZERO_TWO_LEFT(idx_start:end);
    LFP_Right = data.neural.combined_data_table.TD_Other_ZERO_TWO_RIGHT(idx_start:end);
    fprintf('Using TD_Other_ZERO_TWO channels\n');
elseif any(contains(varNames, 'TD_Other_ONE_THREE_LEFT'))
    LFP_Left = data.neural.combined_data_table.TD_Other_ONE_THREE_LEFT(idx_start:end);
    LFP_Right = data.neural.combined_data_table.TD_Other_ONE_THREE_RIGHT(idx_start:end);
    fprintf('Using TD_Other_ONE_THREE channels\n');
else
    error('Could not find LFP channels. Available variables: %s', strjoin(varNames, ', '));
end

% demean the LFP data by subtracting the mean, ignoring NaN values
LFP_demeaned_Left = LFP_Left - mean(LFP_Left, 'omitnan');
LFP_demeaned_Right = LFP_Right - mean(LFP_Right, 'omitnan');

%replace Nan by 0
LFP_demeaned_Left(isnan(LFP_demeaned_Left)) = 0;
LFP_demeaned_Right(isnan(LFP_demeaned_Right)) = 0;

% timestamps (every 4mseg)
t = ((1/data.neural.fs): (1/data.neural.fs): (length(LFP_demeaned_Left)/data.neural.fs));

LFP_r = LFP_demeaned_Right;
LFP_l = LFP_demeaned_Left;
theta_L = bandpass(LFP_demeaned_Left, [4 8], data.neural.fs);
theta_R = bandpass(LFP_demeaned_Right, [4 8], data.neural.fs);

%detect when the task starts
task_flag = data.behavior.first_event_unix_time;


%% compute spectrograms - full data

window = round(.5*data.neural.fs);
noverlap = round(.75*window);
nfft = max(256, 2^nextpow2(window));

[S_Left, F_Left, T_Left] = spectrogram(LFP_demeaned_Left, hamming(window), noverlap, nfft, data.neural.fs, 'yaxis');
[S_Right, F_Right, T_Right] = spectrogram(LFP_demeaned_Right, hamming(window), noverlap, nfft, data.neural.fs, 'yaxis');

log_S_Left = 10 * log10(abs(S_Left));
log_S_Right = 10 * log10(abs(S_Right));

% fixed color scale 0-30 dB
clim_min = 0;
clim_max = 30;

% limit frequencies to 0-30 Hz
freq_mask = F_Left <= 30;
F_Left_display = F_Left(freq_mask);
F_Right_display = F_Right(freq_mask);
log_S_Left_display = log_S_Left(freq_mask, :);
log_S_Right_display = log_S_Right(freq_mask, :);


%% open video
[video_name,video_path] = uigetfile({'*.mp4'},'/Users/agusamilibia/Library/CloudStorage/Box-Box/datalake/AA-56119/AA001/clinic/2025-01-28/Preprocessed videos'); 

v = VideoReader(fullfile(video_path, video_name));

%% output video
output_folder = fullfile(video_path, sub_name_date);

if not(isfolder(output_folder))
    mkdir(output_folder)
end

outputVideo = VideoWriter(fullfile(output_folder, [sub_name_date '_sync_output.mp4']), 'MPEG-4');
outputVideo.Quality = 95;
outputVideo.FrameRate = v.FrameRate;
open(outputVideo);
cleanupObj = onCleanup(@() close(outputVideo));


%figures background always white
set(0, 'DefaultFigureColor', 'w');   
set(0, 'DefaultAxesColor',   'w');
set(0, 'DefaultAxesXColor',  'k');
set(0, 'DefaultAxesYColor',  'k');
set(0, 'DefaultTextColor',   'k');
set(0, 'DefaultAxesFontSize', 14);
set(0, 'DefaultTextFontSize', 14);


f = figure('Units', 'pixels', ...
    'Position', [50, 50, 1920, 1080], ...
    'MenuBar', 'none', ...
    'ToolBar', 'none', ...
    'Resize', 'off', ...
    'Visible', 'off');

set(f, 'Renderer', 'painters');


%% setup the subplots

tiledlayout(5,2, 'TileSpacing', 'compact', 'Padding', 'compact')
ax1 = nexttile([2 2]);
ax2 = nexttile;
ax3 = nexttile;
ax4 = nexttile;
ax5 = nexttile;
ax6 = nexttile;
ax7 = nexttile;

time_window = 5;

% display the first frame as an image in the specified axes
axis(ax1, 'off');
axis(ax1, 'image');
firstFrame = readFrame(v);
p1 = imshow(firstFrame, 'Parent', ax1);

% create time display text at the top - two boxes side by side
% Clock time (left box)
clock_text = annotation('textbox', [0.30, 0.96, 0.18, 0.03], ...
    'String', datestr(datetime_value, 'hh:MM:SS PM'), ...
    'EdgeColor', 'k', ...
    'LineWidth', 2, ...
    'BackgroundColor', 'w', ...
    'FontSize', 18, ...
    'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center',...
    'Color','k');

% Elapsed time (right box)
time_text = annotation('textbox', [0.52, 0.96, 0.18, 0.03], ...
    'String', 'Elapsed: 0.00 s', ...
    'EdgeColor', 'k', ...
    'LineWidth', 2, ...
    'BackgroundColor', 'w', ...
    'FontSize', 18, ...
    'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center',...
    'Color','k');

% left LFP
h2 = plot(ax2, NaN, NaN, '-r', 'LineWidth', 1.5);
ylabel(ax2, 'Amplitude (µV)');
xlabel(ax2, 'Time relative to current (s)');
ylim(ax2, [-30, 30]);
xlim(ax2, [-time_window, time_window]);
grid(ax2, 'on');
title(ax2,'Demeaned LFP – Left Channel', 'FontSize', 16, 'FontWeight', 'bold');

% right LFP
h3 = plot(ax3, NaN, NaN, '-r', 'LineWidth', 1.5);
ylabel(ax3, 'Amplitude (µV)');
xlabel(ax3, 'Time relative to current (s)');
ylim(ax3, [-30, 30]);
xlim(ax3, [-time_window, time_window]);
grid(ax3, 'on');
title(ax3,"Demeaned LFP – Right Channel", 'FontSize', 16, 'FontWeight', 'bold');

% theta wave left
h4 = plot(ax4, NaN, NaN, '-b', 'LineWidth', 1.5);
ylabel(ax4, 'Amplitude (µV)');
xlabel(ax4, 'Time relative to current (s)');
ylim(ax4, [-30, 30]);
xlim(ax4, [-time_window, time_window]);
grid(ax4, 'on');
title(ax4,'Theta wave - Left Channel', 'FontSize', 16, 'FontWeight', 'bold');

% theta wave right
h5 = plot(ax5, NaN, NaN, '-b', 'LineWidth', 1.5);
ylabel(ax5, 'Amplitude (µV)');
xlabel(ax5, 'Time relative to current (s)');
ylim(ax5, [-30, 30]);
xlim(ax5, [-time_window, time_window]);
grid(ax5, 'on');
title(ax5,'Theta wave - Right Channel', 'FontSize', 16, 'FontWeight', 'bold');

% spectrograms with fixed color scale
sh1 = imagesc('Parent', ax6, 'CData', [], 'XData', [-time_window time_window], 'YData', F_Left_display);
title(ax6, 'Left Spectrogram', 'FontSize', 16, 'FontWeight', 'bold'); 
axis(ax6, 'xy'); 
ylabel(ax6, 'Frequency (Hz)');
xlabel(ax6, 'Time relative to current (s)');
ylim(ax6, [0, 30]);
xlim(ax6, [-time_window, time_window]);
cb1 = colorbar(ax6);
cb1.Label.String = 'Power (dB)';
cb1.Label.Color = 'k';
cb1.Color = 'k';   
clim(ax6, [clim_min clim_max]);

sh2 = imagesc('Parent', ax7, 'CData', [], 'XData', [-time_window time_window], 'YData', F_Right_display);
title(ax7, 'Right Spectrogram', 'FontSize', 16, 'FontWeight', 'bold'); 
axis(ax7, 'xy'); 
ylabel(ax7, 'Frequency (Hz)');
xlabel(ax7, 'Time relative to current (s)');
ylim(ax7, [0, 30]);
xlim(ax7, [-time_window, time_window]);
cb2 = colorbar(ax7);
cb2.Label.Color = 'k';
cb2.Color = 'k';  
cb2.Label.String = 'Power (dB)';
clim(ax7, [clim_min clim_max]);

vline2 = xline(ax2, 0, '-k', 'LineWidth', 2);
vline3 = xline(ax3, 0, '-k', 'LineWidth', 2);
vline4 = xline(ax4, 0, '-k', 'LineWidth', 2);
vline5 = xline(ax5, 0, '-k', 'LineWidth', 2);
vline6 = xline(ax6, 0, '-k', 'LineWidth', 2);
vline7 = xline(ax7, 0, '-k', 'LineWidth', 2);

linkaxes([ax2, ax3, ax4, ax5], 'xy');


%% animate

frame_count = 0;

max_time = 10; % JUST FOR TESTING
while hasFrame(v) && v.CurrentTime <= max_time
    
    vidFrame = readFrame(v);
    video_time = v.CurrentTime;
    neural_time = video_time; % Update neural time to current video time

    
    set(p1, 'CData', vidFrame);
    ax1.Visible = 'off';

    % update time display
    current_clock_time = datetime_value + seconds(neural_time);
    set(clock_text, 'String', datestr(current_clock_time, 'hh:MM:SS PM'));
    set(time_text, 'String', sprintf('Elapsed: %.2f s', neural_time));

    % select neural data window around current time
    time_range = (t >= neural_time - time_window) & (t <= neural_time + time_window);
    
    % convert to relative time
    t_rel = t(time_range) - neural_time;
    
    % video update
    set(p1, 'CData', vidFrame);
    ax1.Visible = 'off';


    % update raw LFP plots
    set(h2, 'XData', t_rel, 'YData', LFP_l(time_range));
    set(h3, 'XData', t_rel, 'YData', LFP_r(time_range));

    % update theta-band plots
    set(h4, 'XData', t_rel, 'YData', theta_L(time_range));
    set(h5, 'XData', t_rel, 'YData', theta_R(time_range));

    % update spectrograms - continuous scroll
    spec_time_range = (T_Left >= neural_time - time_window) & (T_Left <= neural_time + time_window);
    
    if sum(spec_time_range) > 0
        spec_data_left = log_S_Left_display(:, spec_time_range);
        spec_data_right = log_S_Right_display(:, spec_time_range);
        spec_times = T_Left(spec_time_range);
        
        % convert to relative time
        spec_times_rel = spec_times - neural_time;
        
        set(sh1, 'CData', spec_data_left, ...
                 'XData', [spec_times_rel(1), spec_times_rel(end)], ...
                 'YData', F_Left_display);
        set(sh2, 'CData', spec_data_right, ...
                 'XData', [spec_times_rel(1), spec_times_rel(end)], ...
                 'YData', F_Right_display);
    end

    frame_count = frame_count + 1;
    % debug message: compare video and neural times
    fprintf('Time: %.2fs | Progress: %.1f%% | Frame: %d\n', ...
        neural_time, 100*video_time/v.Duration, frame_count);

    drawnow;
    frame = getframe(f);
    if frame_count == 1
        expected_size = size(frame.cdata);
    elseif ~isequal(size(frame.cdata), expected_size)
        frame.cdata = imresize(frame.cdata, [expected_size(1), expected_size(2)]);
    end
    writeVideo(outputVideo, frame);

end

close(outputVideo);
fprintf('✅ Video generation complete!\n');

%% add audio

output_video   = fullfile(output_folder, [sub_name_date '_sync_output.mp4']);
original_video = fullfile(video_path, video_name);
final_video    = fullfile(output_folder, [sub_name_date '_sync_with_audio.mp4']);

if exist(final_video, 'file')
    delete(final_video);
end

cmd = sprintf('"%s" -y -i "%s" -i "%s" -c:v libx264 -c:a aac -map 0:v:0 -map 1:a:0 -shortest "%s"', ...
              ffmpeg_path, output_video, original_video, final_video);

fprintf('⏳ Adding audio with FFmpeg...\n');
[status, cmdout] = system(cmd);

if status == 0
    fprintf('✅ Final video created with audio: %s\n', final_video);
    delete(output_video);
else
    fprintf('❌ FFmpeg failed:\n%s\n', cmdout);
    error('Check paths or command syntax.');
end

elapsed_time = toc;
fprintf('\n⏱️  Total elapsed time: %.1f seconds (%.1f minutes)\n', elapsed_time, elapsed_time/60);    