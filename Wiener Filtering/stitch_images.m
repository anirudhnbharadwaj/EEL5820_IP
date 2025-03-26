% stitch_images.m
baseDir = pwd;
inputDir = fullfile(baseDir, 'InputImages');
outputDir = fullfile(baseDir, 'Output');
resultsDir = fullfile(outputDir, 'Results');

if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

% List of imags
images = {
    'blurred_lena_av9', 'avg_9', {'InputImages/blurred_lena_av9.bmp', ...
        'Output/DeblurredImages/Direct/deblurred_blurred_lena_av9_SNR0.0001.png', ...
        'Output/DeblurredImages/Sequential/deblurred_blurred_lena_av9_seq_SNR0.0001.png'};
    'blurred_lena_av19', 'avg_19', {'InputImages/blurred_lena_av19.bmp', ...
        'Output/DeblurredImages/Direct/deblurred_blurred_lena_av19_SNR0.0001.png', ...
        'Output/DeblurredImages/Sequential/deblurred_blurred_lena_av19_seq_SNR0.0001.png'};
    'blurred_lena_m33_29', 'motion_33_29', {'InputImages/blurred_lena_m33_29.bmp', ...
        'Output/DeblurredImages/Direct/deblurred_blurred_lena_m33_29_SNR0.0001.png', ...
        'Output/DeblurredImages/Mismatched/deblurred_blurred_lena_m33_29_theta34_SNR0.0001.png'};
    'blurred_lena_m135_19', 'motion_135_19', {'InputImages/blurred_lena_m135_19.bmp', ...
        'Output/DeblurredImages/Direct/deblurred_blurred_lena_m135_19_SNR0.0001.png', ...
        'Output/DeblurredImages/Mismatched/deblurred_blurred_lena_m135_19_theta136_SNR0.0001.png'};
    'blurred_lena_g_11_3', 'gauss_11_3', {'InputImages/blurred_lena_g_11_3.bmp', ...
        'Output/DeblurredImages/Direct/deblurred_blurred_lena_g_11_3_SNR0.0001.png'};
    'blurred_lena_g_19_7', 'gauss_19_7', {'InputImages/blurred_lena_g_19_7.bmp', ...
        'Output/DeblurredImages/Direct/deblurred_blurred_lena_g_19_7_SNR0.0001.png'};
};

for i = 1:size(images, 1)
    imgBase = images{i, 1};
    outputName = images{i, 2};
    imgPaths = images{i, 3};
    
    imgs = cellfun(@imread, imgPaths, 'UniformOutput', false);
    
    heights = cellfun(@(x) size(x, 1), imgs);
    maxHeight = max(heights);
    for j = 1:length(imgs)
        if size(imgs{j}, 1) < maxHeight
            imgs{j} = padarray(imgs{j}, [maxHeight - size(imgs{j}, 1), 0], 255, 'post');
        end
    end
    
    % Stitch 
    stitched = cat(2, imgs{:});
    
    % Save
    imwrite(stitched, fullfile(resultsDir, [outputName, '.png']));
end
disp('Image stitching complete.');