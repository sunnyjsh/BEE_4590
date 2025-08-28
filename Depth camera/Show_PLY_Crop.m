% Clear workspace and command window
clear;
clc;

% --- User Configuration ---
% Set the filename of your PLY file.
% Make sure this file is in the same directory as the script,
% or provide the full path to the file.
% filename = 'manual_depth_20250827-104525_raw.ply';
filename = 'manual_depth_20250827-104525_segmented.ply';

% --- Main Script ---
try
    % Read the point cloud data from the .ply file
    % pcread is part of the Computer Vision Toolbox.
    disp(['Attempting to read ' filename '...']);
    ptCloud = pcread(filename);
    disp('File read successfully.');

    % --- 1. Display the original point cloud ---
    % Create a new figure to display the point cloud
    figure;
    % pcshow visualizes the 3D point cloud.
    disp('Displaying the original point cloud...');
    pcshow(ptCloud);
    % Add labels and a title for better visualization
    title('Original Point Cloud');
    xlabel('X-axis');
    ylabel('Y-axis');
    zlabel('Z-axis');
    
    % --- 2. Crop and display the point cloud ---
    % Define the region of interest (ROI) for the Z-axis
    zRange = [-1, 0];
    roi = [-1, 0.5; -1, 0.5; -1, 0]; % No limits on X and Y

    % Find the indices of the points within the defined ROI
    indices = findPointsInROI(ptCloud, roi);

    % Select the points within the ROI to create a new, cropped point cloud
    ptCloudCropped = select(ptCloud, indices);
    disp(['Cropped point cloud to Z-range: [' num2str(zRange(1)) ', ' num2str(zRange(2)) ']']);

    % Create a second figure for the cropped point cloud
    figure;
    % Display the cropped point cloud
    disp('Displaying the cropped point cloud...');
    pcshow(ptCloudCropped);
    % Add labels and a title for the cropped visualization
    title('Cropped Point Cloud (-1 < Z < 0)');
    xlabel('X-axis');
    ylabel('Y-axis');
    zlabel('Z-axis');
    
    disp('Visualization complete.');

catch ME
    % Handle potential errors, such as file not found or invalid format
    fprintf(2, 'An error occurred:\n');
    fprintf(2, '%s\n', ME.message);
    disp('Please make sure the file exists and is a valid PLY file.');
    disp('Also, ensure you have the Computer Vision Toolbox installed.');
end
