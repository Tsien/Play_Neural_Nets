%==========================================================================
%Description:
%       sigmoid function
%Input:
%       x : data
%Output:
%       y : the value of sigmoid function
%==========================================================================
function y = Sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

