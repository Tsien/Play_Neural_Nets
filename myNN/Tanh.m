%==========================================================================
%Description:
%       Tanh function
%Input:
%       x : data
%Output:
%       y : the value of Tanh function
%==========================================================================
function y = Tanh(x)
    y = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));
end
