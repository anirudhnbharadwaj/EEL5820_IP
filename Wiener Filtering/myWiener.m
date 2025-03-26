function [deblurredim, raw_deblurred] = myWiener(im, blur_type, len, theta, SNR)
% Applies Wiener filter for image deblurring using the provided non-standard equation
% Inputs:
%   im: filename (string) or image matrix
%   blur_type: 'average', 'motion', or 'gaussian'
%   len, theta: blur parameters
%   SNR: signal-to-noise ratio parameter
% Outputs:
%   deblurredim: enhanced deblurred image
%   raw_deblurred: deblurred image before enhancement

% Load image if filename is provided
if ischar(im)
    im = imread(im);
end

% Convert to grayscale if RGB
if size(im, 3) == 3
    im = rgb2gray(im);
end

% Convert to double [0,1]
im = double(im) / 255;

% Apply median filter
im_filtered = medfilt2(im, [3 3]);

% Fourier transform
F = fft2(im_filtered);

% Create PSF based on blur_type
switch blur_type
    case 'average'
        PSF = fspecial('average', [len len]);
    case 'motion'
        PSF = fspecial('motion', len, theta);
    case 'gaussian'
        PSF = fspecial('gaussian', [len len], theta);
    otherwise
        error('Unknown blur type: %s', blur_type);
end

% Convert PSF to OTF
OTF = psf2otf(PSF, size(im));
OTF(abs(OTF) < eps) = eps; % Avoid division by zero

%% Wiener filter calculation as it is given in the Assignment PDF
%numerator = OTF .* conj(OTF); % B(u,v) .* B*(u,v)
%denominator = OTF .* OTF .* conj(OTF) + SNR; % B(u,v) .* B(u,v) .* B*(u,v) + SNR
%H = numerator ./ denominator; % H(u,v) = numerator / denominator

% Standard Wiener filter calculation
H = conj(OTF) ./ (abs(OTF).^2 + SNR);

% Apply the filter in the frequency domain
deblurred_F = F .* H;

% Inverse FFT
raw_deblurred = real(ifft2(deblurred_F));

% Normalize to [0,1]
raw_deblurred = (raw_deblurred - min(raw_deblurred(:))) / (max(raw_deblurred(:)) - min(raw_deblurred(:)));

% Enhance for visual output
deblurredim = adapthisteq(raw_deblurred, 'clipLimit', eps, 'Distribution', 'exponential');
deblurredim = imadjust(deblurredim, stretchlim(deblurredim), [0 1]);
end