% wiener_deblur_all.m
% Applies Wiener filter to all test images, saves results as PNG.

clear;
clc;

% Folder structure
baseDir = pwd;
inputDir = fullfile(baseDir, 'InputImages');
outputDir = fullfile(baseDir, 'Output2');
deblurDir = fullfile(outputDir, 'DeblurredImages');
directDir = fullfile(deblurDir, 'Direct');
seqDir = fullfile(deblurDir, 'Sequential');
mismatchDir = fullfile(deblurDir, 'Mismatched');
resultsDir = fullfile(outputDir, 'Results');

% Create output directories 
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
if ~exist(deblurDir, 'dir'), mkdir(deblurDir); end
if ~exist(directDir, 'dir'), mkdir(directDir); end
if ~exist(seqDir, 'dir'), mkdir(seqDir); end
if ~exist(mismatchDir, 'dir'), mkdir(mismatchDir); end
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

% Load original image
original = imread(fullfile(inputDir, 'lena.bmp'));
original = rgb2gray(original);
original = double(original) / 255;

% Blurred images
blurred_images = {
    'blurred_lena_av9.bmp', ...
    'blurred_lena_av19.bmp', ...
    'blurred_lena_m33_29.bmp', ...
    'blurred_lena_m135_19.bmp', ...
    'blurred_lena_g_11_3.bmp', ...
    'blurred_lena_g_19_7.bmp'
};

% SNR values to experiment with
SNR_values = [0.1, 0.01, 0.001, 0.0001];

% Initialize results table
results = table('Size', [0, 8], ...
    'VariableTypes', {'string', 'string', 'double', 'string', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'Image', 'BlurType', 'SNR', 'Method', 'PSNR_raw', 'SSIM_raw', 'PSNR_enh', 'SSIM_enh'});

% Process image
for i = 1:length(blurred_images)
    imname = blurred_images{i};
    imPath = fullfile(inputDir, imname);
    [blur_type, len, theta] = parse_filename(imname);
    [~, name, ~] = fileparts(imname);

    for j = 1:length(SNR_values)
        SNR = SNR_values(j);

        % Direct Deblurring
        [enhanced, raw] = myWiener(imPath, blur_type, len, theta, SNR);
        save_name = sprintf('deblurred_%s_SNR%.4f.png', name, SNR); 
        imwrite(enhanced, fullfile(directDir, save_name));
        psnr_raw = psnr(raw, original);
        ssim_raw = ssim(raw, original);
        psnr_enh = psnr(enhanced, original);
        ssim_enh = ssim(enhanced, original);
        results = [results; {imname, blur_type, SNR, 'direct', psnr_raw, ssim_raw, psnr_enh, ssim_enh}];

        % Sequential Deblurring for Average Blur
        if strcmp(blur_type, 'average')
            [~, raw1] = myWiener(imPath, 'motion', len, 0, SNR);
            [enhanced_seq, raw_seq] = myWiener(raw1, 'motion', len, 90, SNR);
            save_name_seq = sprintf('deblurred_%s_seq_SNR%.4f.png', name, SNR); 
            imwrite(enhanced_seq, fullfile(seqDir, save_name_seq));
            psnr_seq_raw = psnr(raw_seq, original);
            ssim_seq_raw = ssim(raw_seq, original);
            psnr_seq_enh = psnr(enhanced_seq, original);
            ssim_seq_enh = ssim(enhanced_seq, original);
            results = [results; {imname, blur_type, SNR, 'sequential', psnr_seq_raw, ssim_seq_raw, psnr_seq_enh, ssim_seq_enh}];
        end

        % Parameter Mismatch for Motion Blur
        if strcmp(blur_type, 'motion')
            % Theta + 1
            [enhanced_off, raw_off] = myWiener(imPath, blur_type, len, theta + 1, SNR);
            save_name_off = sprintf('deblurred_%s_theta%d_SNR%.4f.png', name, theta + 1, SNR); 
            imwrite(enhanced_off, fullfile(mismatchDir, save_name_off));
            psnr_off_raw = psnr(raw_off, original);
            ssim_off_raw = ssim(raw_off, original);
            psnr_off_enh = psnr(enhanced_off, original);
            ssim_off_enh = ssim(enhanced_off, original);
            results = [results; {imname, blur_type, SNR, 'theta+1', psnr_off_raw, ssim_off_raw, psnr_off_enh, ssim_off_enh}];

            % Theta - 1
            [enhanced_off2, raw_off2] = myWiener(imPath, blur_type, len, theta - 1, SNR);
            save_name_off2 = sprintf('deblurred_%s_theta%d_SNR%.4f.png', name, theta - 1, SNR); 
            imwrite(enhanced_off2, fullfile(mismatchDir, save_name_off2));
            psnr_off2_raw = psnr(raw_off2, original);
            ssim_off2_raw = ssim(raw_off2, original);
            psnr_off2_enh = psnr(enhanced_off2, original);
            ssim_off2_enh = ssim(enhanced_off2, original);
            results = [results; {imname, blur_type, SNR, 'theta-1', psnr_off2_raw, ssim_off2_raw, psnr_off2_enh, ssim_off2_enh}];
        end
    end
end

% Save results as CSV
csvPath = fullfile(resultsDir, 'wiener_deblur_results.csv');
writetable(results, csvPath);
disp(['Results saved to: ', csvPath]);

% Convert string arrays to char arrays 
resultsCell = table2cell(results);
for row = 1:size(resultsCell, 1)
    for col = 1:size(resultsCell, 2)
        if isstring(resultsCell{row, col})
            resultsCell{row, col} = char(resultsCell{row, col});
        end
    end
end

% Save table as an image
fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);
t = uitable(fig, 'Data', resultsCell, 'ColumnName', results.Properties.VariableNames, ...
    'Position', [20, 20, 1150, 750]);
t.FontSize = 8;
imgPath = fullfile(resultsDir, 'wiener_deblur_results.png');
saveas(fig, imgPath);
close(fig);
disp(['Table image saved to: ', imgPath]);

disp('Processing complete.');