%==========================================================================
%Description:
%      extract images from MNIST
%Input:
%      file : the raw data
%Output:
%      img  : a 28x28x#images matrix, already rescale to [0,1] double
%==========================================================================
%img = readMNISTImages('train-images.idx3-ubyte');
function img = readMNISTImages(file)
    
    fp = fopen(file, 'rb');
    if (fp == -1)
        display( ['Could not open ', file, '']);
    end

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    if (magic ~= 2051)
        display( ['Bad magic number in ', file, '']);
    end

    numImgs = fread(fp, 1, 'int32', 0, 'ieee-be');
    rows = fread(fp, 1, 'int32', 0, 'ieee-be');
    cols = fread(fp, 1, 'int32', 0, 'ieee-be');

    img = fread(fp, inf, 'unsigned char');
    img = reshape(img, cols, rows, numImgs);
    img = permute(img,[2 1 3]);

    fclose(fp);

    img = reshape(img, size(img, 1) * size(img, 2), size(img, 3));
    % rescale to [0,1]
    img = double(img) / 255;

end