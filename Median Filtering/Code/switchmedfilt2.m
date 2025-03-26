function denoisedim = switchmedfilt2(noisyim, wsize)

% 
% This function takes as input a noisy image (noisyim) and the window size
% (wsize) to perform Switching Median Filtering on the noisy image, and
% outputs the denoised image (denoisedim) with the resulting value of the
% signal-to-noise (snrval)
%

if mod(wsize, 2) == 0
    error('Window size (wsize) must be an odd integer.');
end

pad = floor(wsize/2);
padded = padarray(noisyim, [pad pad], 'replicate', 'both');
[rows, cols] = size(noisyim);
denoisedim = zeros(rows, cols);

for i = 1:rows
    for j = 1:cols
        i_pad = i + pad;
        j_pad = j + pad;
        window = padded(i_pad - pad:i_pad + pad, j_pad - pad:j_pad + pad);
        vec = window(:);
        var_incl = var(vec);
        center = (numel(vec) + 1)/2;
        vec_excl = vec([1:center-1, center+1:end]);
        var_excl = var(vec_excl);
        
        if var_incl > var_excl
            denoisedim(i, j) = median(vec);
        else
            denoisedim(i, j) = noisyim(i, j);
        end
    end
end
end