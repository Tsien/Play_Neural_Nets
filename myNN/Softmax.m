%==========================================================================
%Description:
%       Softmax function
%Input:
%       x : Softmax
%Output:
%       y : the result of Softmax function
%==========================================================================
function y = Softmax(x)
    x = bsxfun(@minus, x, max(x, [], 2));%to avoid overflow, NaN
    y = exp(x);
    y = bsxfun(@rdivide, y, y * ones(size(y, 2), 1));
end

