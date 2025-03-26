function snrvalue = mysnr(refim, noisyim)
%
% This function returns the SNR value computed for a noisy image (noisyim)
% with respect to a reference image (refim)
%
%

refsig=double(refim(:));
noise=refsig-double(noisyim(:));
signalPow = sum(refsig.^2);
noisePow  = sum(noise.^2);
snrvalue = 10 * log10(signalPow / noisePow);

end