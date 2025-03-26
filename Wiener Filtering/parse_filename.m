function [blur_type, len, theta] = parse_filename(imname)
% Parses the filename to determine blur type and parameters
imname_lower = lower(imname);
if contains(imname_lower, 'av')
    blur_type = 'average';
    tokens = regexp(imname, 'av(\d+)', 'tokens');
    if isempty(tokens)
        error('Average blur pattern ''av(\\d+)'' not found in filename: %s', imname);
    end
    len = str2double(tokens{1}{1});
    theta = NaN;
elseif contains(imname_lower, 'm') && ~isempty(regexp(imname_lower, 'm\d+_\d+', 'once'))
    blur_type = 'motion';
    tokens = regexp(imname, 'm(\d+)_(\d+)', 'tokens');
    if isempty(tokens)
        error('Motion blur pattern ''m(\\d+)_(\\d+)'' not found in filename: %s', imname);
    end
    theta = str2double(tokens{1}{1});
    len = str2double(tokens{1}{2});
elseif contains(imname_lower, 'g_')
    blur_type = 'gaussian';
    tokens = regexp(imname, 'g_(\d+)_(\d+)', 'tokens');
    if isempty(tokens)
        error('Gaussian blur pattern ''g_(\\d+)_(\\d+)'' not found in filename: %s', imname);
    end
    len = str2double(tokens{1}{1});
    theta = str2double(tokens{1}{2});
else
    error('Unknown blur type in image name: %s', imname);
end
end