%==========================================================================
%Description:
%       Softmax function
%Input:
%       x : Softmax
%Output:
%       y : the result of Softmax function
%==========================================================================
function y = Softmax(x)
    y = exp(x);
    y = bsxfun(@rdivide, y, y * ones(size(y, 2), 1));
end

