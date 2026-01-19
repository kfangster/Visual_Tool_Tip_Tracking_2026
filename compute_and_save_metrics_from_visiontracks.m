%%% Code to calculate tool motion metrics from CV-derived trajectory data (2D)
% Incorporates Trim and PCHIP interpolation for robust processing

fprintf("Clearing workspace and command window...\n");
pause(10); 
clc; clear all; 

TASK            = "T1";           % 'T1', 'T2', 'T3'; corresponds to Suturing, Knot tying, Needle driving
KEYPOINT_TYPE   = 'ToolJawBase';  % 'ToolTip', 'ToolJawBase', 'Centroid'
MODEL_TYPE      = 'YOLO';           % 'YOLO', 'MFCNet'
SAVE_DIR        = "./computed_metrics/";
OUTPUT_FNAME = [char(MODEL_TYPE), '_metrics_cv2d_', strrep(char(TASK), '_', ''), '_', char(KEYPOINT_TYPE), '.csv'];


column_names = {
    'File Name', 'Participant ID', 'Participant Trial Number', ...  % Metadata
    'Time (s)', 'Tool 1 Idle Time', 'Tool 2 Idle Time', ...         % Time-related metrics
    'Tool 1 Path Length', 'Tool 2 Path Length', 'Tool 1 RMS', 'Tool 2 RMS', ... % Zero-th order metrics
    'Tool 1 Average Velocity', 'Tool 2 Average Velocity', 'Tool 1 SPARC', 'Tool 2 SPARC', ... % First order metrics
    'Tool 1 Mean Acceleration', 'Tool 2 Mean Acceleration', ... % Second order metrics
    'Tool 1 LDJ', 'Tool 2 LDJ', ...                             % Third order metrics
    'Tool 1 Missing Data (%)', 'Tool 2 Missing Data (%)' ...    % Auxiliary metrics
};

if strcmp(MODEL_TYPE, 'MFCNet')
    % DATA_DIR = './model_yolo_v8l_run1/';
    % DATA_DIR = [DATA_DIR, 'video_tracking_results/'];
elseif strcmp(MODEL_TYPE, 'YOLO')
    DATA_DIR = './model_yolo_v8l_run1/';
    % DATA_DIR = [DATA_DIR, 'video_tracking_results_yolo/'];
else
    error('Invalid MODEL_TYPE specified. Choose either "MFCNet" or "YOLO".');
end
filePattern = fullfile(DATA_DIR, ['P*_', char(TASK), '_R*_EndoRight_tracked.csv']);

OUTPUT_FPATH    = fullfile(SAVE_DIR, OUTPUT_FNAME);

% metric computation-related hyperparameters
pixel_size      = [1440, 1080];       % Size of frames in pixels for normalization
Fs_all          = 30;               % Sampling frequency
T               = 1 / Fs_all;       % Sampling period
order           = 3;                % Savitzky-Golay filter order
window          = 21;               % Savitzky-Golay filter window size
SPARC_parameters = [0.05, 20, 4];   % SPARC parameters
vel_threshold   = 0.05;             % Velocity threshold for idle time calculation (b/w 2 consecutive frames)

files = dir(filePattern); 
num_files = length(files);
results = cell(num_files, length(column_names));

for n = 1:num_files
    % Extract participant ID and trial number from file name 
    filename = files(n).name;
    parts = split(filename, '_');
    participant_id = parts{1};
    trial_number = str2double(parts{3}(2:end));

    % Read and preprocess file (normalize data)
    data = readmatrix(fullfile(DATA_DIR, filename));
    fprintf('Processing file %s\n', filename);
    if strcmp(KEYPOINT_TYPE, 'ToolTip')
        data = data(:, 1:8); 
        data(:, 2:9) = data;  
        for l = 2:2:9
            data(:, l) = data(:, l) / pixel_size(1); % Normalize x-coordinates
        end
        for l = 3:2:9
            data(:, l) = data(:, l) / pixel_size(2); % Normalize y-coordinates
        end
    elseif strcmp(KEYPOINT_TYPE, 'ToolJawBase')
        data = data(:, 9:12);
        data(:, 2:5) = data;
        for l = 2:2:5
            data(:, l) = data(:, l) / pixel_size(1); % Normalize x-coordinates
        end
        for l = 3:2:5
            data(:, l) = data(:, l) / pixel_size(2); % Normalize y-coordinates
        end
    elseif strcmp(KEYPOINT_TYPE, 'Centroid')
        data = data(:, 13:16);
        data(:, 2:5) = data;
        for l = 2:2:5
            data(:, l) = data(:, l) / pixel_size(1); % Normalize x-coordinates
        end
        for l = 3:2:5
            data(:, l) = data(:, l) / pixel_size(2); % Normalize y-coordinates
        end
    end
    data(:, 1) = 0:T:size(data, 1) * T - T;     % Add Time vector

    % Get trajectory (of both tools) to compute metrics
    trajectory = zeros(size(data, 1), 4); % Initialize trajectories
    if strcmp(KEYPOINT_TYPE, 'ToolTip')
        trajectory(:, 1:2) = (data(:, 2:3) + data(:, 4:5)) / 2; % Tool 1 (x,y)
        trajectory(:, 3:4) = (data(:, 6:7) + data(:, 8:9)) / 2; % Tool 2 (x,y) 
    elseif strcmp(KEYPOINT_TYPE, 'ToolJawBase')
        trajectory = data(:, 2:5); % Use ToolJawBase directly
    elseif strcmp(KEYPOINT_TYPE, 'Centroid')
        trajectory = data(:, 2:5); % Use Centroid directly 
    end

    tool1_missing_data = (sum(isnan(trajectory(:,1)))*T)/(data(end, 1))*100; % Calculate missing data percentage for Tool 1
    tool2_missing_data = (sum(isnan(trajectory(:,3)))*T)/(data(end, 1))*100; % Calculate missing data percentage for Tool 2

    % Compute RMS spread of trajectories (on raw data before trimming/interpolation)
    x1 = trajectory(:, 1); y1 = trajectory(:, 2);
    valid_idx1 = ~isnan(x1) & ~isnan(y1);
    x1_valid = x1(valid_idx1); y1_valid = y1(valid_idx1);
    mean_x1 = mean(x1_valid); mean_y1 = mean(y1_valid);
    tool1_rms = sqrt(mean((x1_valid - mean_x1).^2 + (y1_valid - mean_y1).^2));

    x2 = trajectory(:, 3); y2 = trajectory(:, 4);
    valid_idx2 = ~isnan(x2) & ~isnan(y2);
    x2_valid = x2(valid_idx2); y2_valid = y2(valid_idx2);
    mean_x2 = mean(x2_valid); mean_y2 = mean(y2_valid);
    tool2_rms = sqrt(mean((x2_valid - mean_x2).^2 + (y2_valid - mean_y2).^2));
    
    %==========================================================================
    % REVISED & CORRECTED PROCESSING BLOCK FOR METRICS SCRIPT
    %
    % Implements independent processing for each tool to maximize data usage
    % for individual tool metrics, as per your valid correction.
    %==========================================================================

    % NOTE: This replaces the previous "Trim, PCHIP, and Derivative" block.

    % We will calculate metrics for each tool in a loop to handle them independently.
    all_tool_metrics = struct();
    tool_data_columns = {1:2, 3:4}; % Columns for Tool 1 (L) and Tool 2 (R)

    for tool_idx = 1:2
        tool_traj_raw = trajectory(:, tool_data_columns{tool_idx});

        % Step 1: Trim this specific tool's data
        valid_rows = any(~isnan(tool_traj_raw), 2);
        first_valid_idx = find(valid_rows, 1, 'first');
        last_valid_idx  = find(valid_rows, 1, 'last');

        % Handle case where one tool has no data at all
        if isempty(first_valid_idx)
            fprintf('Warning: No valid data for Tool %d in file %s.\n', tool_idx, filename);
            % Store NaNs for this tool's metrics
            all_tool_metrics(tool_idx).path_length = nan;
            all_tool_metrics(tool_idx).rms = nan;
            all_tool_metrics(tool_idx).av = nan;
            all_tool_metrics(tool_idx).SPARC = nan;
            all_tool_metrics(tool_idx).ma = nan;
            all_tool_metrics(tool_idx).LDJ = nan;
            all_tool_metrics(tool_idx).IT = nan;
            all_tool_metrics(tool_idx).time = 0;
            continue; % Go to the next tool
        end

        tool_traj_trimmed = tool_traj_raw(first_valid_idx:last_valid_idx, :);

        % Step 2: PCHIP Interpolate the trimmed data
        tool_traj_clean = tool_traj_trimmed;
        for i = 1:size(tool_traj_clean, 2)
            tool_traj_clean(:,i) = fillmissing(tool_traj_clean(:,i), 'pchip');
        end

        % Step 3: Derive clean velocity and acceleration
        velocity = diff(tool_traj_clean) / T;
        acceleration = diff(velocity) / T;

        % Step 4: Calculate all metrics for THIS tool
        all_tool_metrics(tool_idx).path_length = sum(sqrt(sum(diff(tool_traj_clean).^2, 2)), 'omitnan');
        
        % RMS must be calculated on original, untrimmed data for that tool
        x_raw = tool_traj_raw(:,1); y_raw = tool_traj_raw(:,2);
        valid_pts = ~isnan(x_raw) & ~isnan(y_raw);
        all_tool_metrics(tool_idx).rms = sqrt(mean((x_raw(valid_pts) - mean(x_raw(valid_pts))).^2 + (y_raw(valid_pts) - mean(y_raw(valid_pts))).^2));

        % Filter velocity and compute speed
        velocity_filt = sgolayfilt(velocity, order, window);
        speed = vecnorm(velocity_filt, 2, 2);

        all_tool_metrics(tool_idx).av = mean(speed);
        all_tool_metrics(tool_idx).SPARC = SpectralArcLength(speed, T, SPARC_parameters);
        all_tool_metrics(tool_idx).ma = mean(vecnorm(acceleration, 2, 2));
        all_tool_metrics(tool_idx).LDJ = log_dimensionless_jerk(speed, Fs_all);
        
        % Calculate Idle Time for this tool
        tool_time = size(tool_traj_trimmed, 1) * T;
        idle_frames = sum(speed < vel_threshold);
        all_tool_metrics(tool_idx).IT = 100 * (idle_frames * T) / tool_time;
        all_tool_metrics(tool_idx).time = tool_time;
    end

    % Extract total time from the raw data before trimming (for missing data %)
    total_duration_raw = size(data, 1) * T;

    % Consolidate results for writing to file
    time = max([all_tool_metrics.time]); % Total duration is the max of the two tool durations

    results(n, :) = {
        filename, participant_id, trial_number, ...
        time, all_tool_metrics(1).IT, all_tool_metrics(2).IT, ...
        all_tool_metrics(1).path_length, all_tool_metrics(2).path_length, all_tool_metrics(1).rms, all_tool_metrics(2).rms, ...
        all_tool_metrics(1).av, all_tool_metrics(2).av, all_tool_metrics(1).SPARC, all_tool_metrics(2).SPARC, ...
        all_tool_metrics(1).ma, all_tool_metrics(2).ma, ...
        all_tool_metrics(1).LDJ, all_tool_metrics(2).LDJ, ...
        (sum(isnan(trajectory(:,1)))*T)/total_duration_raw*100, (sum(isnan(trajectory(:,3)))*T)/total_duration_raw*100 ...
    };
end

writecell([column_names; results], OUTPUT_FPATH); 
fprintf('Results saved to %s\n', OUTPUT_FPATH);
