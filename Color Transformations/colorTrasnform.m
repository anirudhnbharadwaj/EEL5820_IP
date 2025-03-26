clear all;
close all;
clc;

function processImagePairs(sourceDir, targetDir, outputDir)
    % Process all image pairs from source and target directories
    % Target provides color characteristics, Source is transformed
    % Inputs:
    %   sourceDir - Directory with source images (s0 to s6)
    %   targetDir - Directory with target images (t0 to t6)
    %   outputDir - Directory to save output images

    % Output directory check
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Get image files
    sourceFiles = [dir(fullfile(sourceDir, 's*.bmp')); dir(fullfile(sourceDir, 's*.jpg'))];
    targetFiles = [dir(fullfile(targetDir, 't*.bmp')); dir(fullfile(targetDir, 't*.jpg'))];

    % Check if numbers match
    if length(sourceFiles) ~= length(targetFiles)
        error('Mismatch in number of source (%d) and target (%d) images.', ...
              length(sourceFiles), length(targetFiles));
    end

    % Process each pair
    for i = 1:length(sourceFiles)
        sourceNum = extractNumber(sourceFiles(i).name);
        targetNum = extractNumber(targetFiles(i).name);

        if sourceNum ~= targetNum
            warning('Mismatched pair: %s and %s. Skipping.', ...
                    sourceFiles(i).name, targetFiles(i).name);
            continue;
        end

        sourcePath = fullfile(sourceDir, sourceFiles(i).name);
        targetPath = fullfile(targetDir, targetFiles(i).name);
        colorTransfer(sourcePath, targetPath, outputDir);
    end
end

function num = extractNumber(filename)
    % Extract number from filename 
    numStr = regexp(filename, '\d+', 'match');
    num = str2double(numStr{1});
end

function colorTransfer(sourcePath, targetPath, outputDir)

    % Load and preprocess images
    [sourceImg, targetImg] = loadAndPreprocessImages(sourcePath, targetPath);
    
    % Perform color transfer
    outputImg = performColorTransfer(sourceImg, targetImg);
    
    % Save and display results
    saveAndDisplayResults(sourceImg, targetImg, outputImg, sourcePath, targetPath, outputDir);
end

function [sourceImg, targetImg] = loadAndPreprocessImages(sourcePath, targetPath)
    % Load and preprocess source and target images
    sourceImg = imread(sourcePath);
    targetImg = imread(targetPath);
    
    % Convert to double 
    sourceImg = double(sourceImg);
    targetImg = double(targetImg);
    
    % Verify range [0, 255] 
    if max(sourceImg(:)) <= 1
        sourceImg = sourceImg * 255;
    end
    if max(targetImg(:)) <= 1
        targetImg = targetImg * 255;
    end
end

function outputImg = performColorTransfer(sourceImg, targetImg)
    % Perform color transfer: Transform source to match target colors
    [m, n, ~] = size(sourceImg); % Source dimensions (to be transformed)
    
    % Step 1: Vectorize images (N × 3)
    S = reshape(sourceImg, [], 3); % Source: N_s × 3 (to be transformed)
    T = reshape(targetImg, [], 3); % Target: N_t × 3 (provides colors)
    
    % Step 2: Compute mean vectors and covariance matrices
    mean_s = mean(S, 1)'; % 3 × 1 (source mean)
    mean_t = mean(T, 1)'; % 3 × 1 (target mean: rt, gt, bt)
    Cs = cov(S); % 3 × 3 (source covariance)
    Ct = cov(T); % 3 × 3 (target covariance)
    
    % Step 3: Perform SVD on covariance matrices
    [Us, Ds, ~] = svd(Cs); % Source
    [Ut, Dt, ~] = svd(Ct); % Target
    
    % Step 4: Construct transformation matrices
    % Target provides the "to" space, Source is the "from" space
    Mt = [eye(3) mean_t; 0 0 0 1]; % Translate to target mean
    Ms = [eye(3) -mean_s; 0 0 0 1]; % Subtract source mean
    Rt = [Ut zeros(3,1); 0 0 0 1]; % Target rotation
    Rs = inv([Us zeros(3,1); 0 0 0 1]); % Inverse source rotation
    Wt = diag([sqrt(diag(Dt))' 1]); % Target scaling
    Ws = diag([1./sqrt(diag(Ds))' 1]); % Inverse source scaling
    
    % Compute transformation matrix F
    % F maps source colors to match target distribution
    F = Mt * Rt * Wt * Ws * Rs * Ms;
    
    % Step 5: Apply transformation to source
    S_aug = [S'; ones(1, size(S, 1))]; % 4 × N_s
    S_new = F * S_aug; % 4 × N_s
    S_transformed = S_new(1:3, :)'; % N_s × 3
    
    % Reshape back to source image dimensions
    outputImg = reshape(S_transformed, m, n, 3);
    
    % Post-process: Clip to [0, 255]
    outputImg = max(0, min(255, outputImg));
    outputImg = uint8(round(outputImg));
end

function saveAndDisplayResults(sourceImg, targetImg, outputImg, sourcePath, targetPath, outputDir)
    % Save and display: source (original), target (unchanged), transformed source
    
    % Extract filenames without extension
    [~, sourceName, ~] = fileparts(sourcePath);
    [~, targetName, ~] = fileparts(targetPath);
    
    % Construct output filenames
    transformedOut = fullfile(outputDir, ['transformed_' sourceName '.jpg']);
    stitchedOut = fullfile(outputDir, ['stitched_' sourceName '_' targetName '.jpg']);
    
    % Save transformed image (source transformed to target colors)
    imwrite(outputImg, transformedOut);
    
    % Create stitched image (resize to same height)
    h = max([size(sourceImg,1), size(targetImg,1), size(outputImg,1)]);
    sourceResized = imresize(uint8(sourceImg), [h NaN]);
    targetResized = imresize(uint8(targetImg), [h NaN]);
    outputResized = imresize(outputImg, [h NaN]);
    stitchedImg = [sourceResized targetResized outputResized];
    imwrite(stitchedImg, stitchedOut);
    
    % Display results
    figure('Name', 'Color Transfer Results', 'NumberTitle', 'off');
    subplot(1, 3, 1); imshow(uint8(sourceImg)); title('Source');
    subplot(1, 3, 2); imshow(uint8(targetImg)); title('Target');
    subplot(1, 3, 3); imshow(outputImg); title('Transformed Source');
end


### Change the source and target directories and run to reproduce the results

sourceDir = '/MATLAB Drive/HW5/source';
targetDir = '/MATLAB Drive/HW5/target';
outputDir = '/MATLAB Drive/HW5/output';
processImagePairs(sourceDir, targetDir, outputDir);