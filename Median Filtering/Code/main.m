%% main.m
% This script loads a test image, adds salt-and-pepper noise, pads the noisy image using padarray, and passes the padded image to switchmedfilt2 to denoise.
% It also compares the results with MATLAB's medfilt2.

inputImagePath = 'cameraman.png'; % test image path

[~, imageName, ~] = fileparts(inputImagePath);
outputFolder = [imageName '_results'];
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

try
    % Read image, convert to grayscale (if needed) and to double precision
    original = im2double(im2gray(imread(inputImagePath)));
catch
    error('Failed to load image. Check the input path.');
end

%% Parameters
wsize = 3;                     % Filter window size
pad = floor(wsize/2);          % Padding width
noiseLevels = 0.1:0.1:0.4;     % Noise densities (from 0.1 to 0.4)
numLevels = length(noiseLevels);

% Preallocate 
snr_switch = zeros(numLevels, 1);
snr_med = zeros(numLevels, 1);

%% Process each noise level
for n = 1:numLevels
    % Create a subfolder for each noise level
    noiseFolder = fullfile(outputFolder, sprintf('noise_%.1f', noiseLevels(n)));
    if ~exist(noiseFolder, 'dir')
        mkdir(noiseFolder);
    end
    
    % salt and pepper noise 
    noisy = imnoise(original, 'salt & pepper', noiseLevels(n));
    
    % Padding the noisy image - symmetric padding.
    paddedNoisy = padarray(noisy, [pad pad], 'symmetric', 'both');
    
    % padded noisy image to switchmedfilt2.
    denoised_switch_padded = switchmedfilt2(paddedNoisy, wsize);
    % Remove the pad image of the same size as original.
    denoised_switch = denoised_switch_padded(pad+1:end-pad, pad+1:end-pad);
    
    % Pad first so that the filtering operation is applied on a padded image.
    paddedNoisy_med = padarray(noisy, [pad pad], 'symmetric', 'both');
    denoised_med_padded = medfilt2(paddedNoisy_med, [wsize wsize]);
    denoised_med = denoised_med_padded(pad+1:end-pad, pad+1:end-pad);
    
    % SNR Calculation
    % Since the output images are unpadded, they have the same dimensions as original.
    snr_switch(n) = mysnr(original, denoised_switch);
    snr_med(n)    = mysnr(original, denoised_med);
    
    imwrite(noisy, fullfile(noiseFolder, 'noisy.jpg'));
    imwrite(denoised_switch, fullfile(noiseFolder, 'denoised_switch.jpg'));
    imwrite(denoised_med, fullfile(noiseFolder, 'denoised_medfilt2.jpg'));
end

%% Save SNR Data to CSV
results = table(noiseLevels', snr_switch, snr_med, ...
    'VariableNames', {'NoiseLevel', 'SNR_SwitchMed', 'SNR_MedFilt2'});
writetable(results, fullfile(outputFolder, 'snr_results.csv'));

%% Display SNR Comparison in a Table (as an image)
fig = figure('Position', [100 100 600 200], 'Color', 'white', 'Visible', 'on');
columnNames = {'Noise Density', 'switchmedfilt2 (dB)', 'medfilt2 (dB)'};
data = [noiseLevels' snr_switch snr_med];
t = uitable(fig, 'Data', data, 'ColumnName', columnNames, ...
    'Position', [20 20 560 160], 'FontSize', 12);
t.ColumnWidth = {100, 150, 150};
drawnow; 
saveas(fig, fullfile(outputFolder, 'snr_comparison_table.png'));
close(fig);

%% Plot SNR vs. Noise Level
figure;
plot(noiseLevels, snr_switch, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;
plot(noiseLevels, snr_med, 'r--x', 'LineWidth', 1.5, 'MarkerSize', 8);
hold off;
legend('switchmedfilt2', 'medfilt2', 'Location', 'southwest');
xlabel('Noise Density');
ylabel('SNR (dB)');
title('SNR Comparison: Adaptive vs. Standard Median Filter');
grid on;
saveas(gcf, fullfile(outputFolder, 'snr_vs_noise.png'));
close;

%% Recursive Filtering Experiment (at 40% Noise)
% Create a high noisy image with noise density 0.4.
noisy_high = imnoise(original, 'salt & pepper', 0.4);
% Pad the high noisy image.
paddedNoisy_high = padarray(noisy_high, [pad pad], 'symmetric', 'both');
current_padded = paddedNoisy_high;
snr_recursive = zeros(10, 1);

for iter = 1:10
    current_padded = switchmedfilt2(current_padded, wsize);
    current_denoised = current_padded(pad+1:end-pad, pad+1:end-pad);
    snr_recursive(iter) = mysnr(original, current_denoised);
end

% Save recursive SNR data to CSV.
recursiveTable = table((1:10)', snr_recursive, ...
    'VariableNames', {'Iteration', 'SNR'});
writetable(recursiveTable, fullfile(outputFolder, 'recursive_snr.csv'));

% Plot recursive SNR evolution.
figure;
plot(1:10, snr_recursive, 'm-s', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('Iteration');
ylabel('SNR (dB)');
title('Recursive Filtering (40% Noise)');
grid on;
saveas(gcf, fullfile(outputFolder, 'recursive_snr_plot.png'));
close;

disp('Processing complete. Results saved in:');
disp(outputFolder);
