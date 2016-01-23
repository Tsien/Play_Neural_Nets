%==========================================================================
%Description:
%       test accuracy by a trained neural networks.
%Input:
%       nn     : neural networks including its parameters and structure
%       test_x : images, already rescale to [0,1] double, 60000X784
%       test_y : labels
%Output:
%       acc    : test accuracy
%==========================================================================
function acc = testNN(nn, test_x, test_y)
    num = size(test_x,1);
    nn = forwardNN(nn, test_x, zeros(num, nn.architecture(end)));
    [tmp, labels] = max(nn.activation{end},[],2);
    [tmp, expected] = max(test_y,[],2);
    error = find(labels ~= expected);    
    acc = 1 - numel(error) / num;
end


