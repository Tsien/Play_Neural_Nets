%==========================================================================
%Description:
%      extract labels from MNIST
%Input:
%      file : the raw data
%Output:
%      label: a #images matrix x 1
%==========================================================================
%test_labels = readMNISTLabels('t10k-labels.idx1-ubyte');
function label = readMNISTLabels(file)
    fp = fopen(file, 'rb');
    if (fp == -1)
        display( ['Could not open ', file, '']);
    end

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    if (magic ~= 2049)
        display( ['Bad magic number in ', file, '']);
    end

    numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

    label = fread(fp, inf, 'unsigned char');

    fclose(fp);

end
