%==========================================================================
%Description:
%       ReLU function
%Input:
%       x : data
%Output:
%       y : the value of ReLU function
%==========================================================================
function y = ReLU(x)
    y = max(0, x);
end