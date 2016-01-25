%==========================================================================
%Description:
%       Funny Tanh function
%Input:
%       x : data
%Output:
%       y : the value of funny tanh function
%==========================================================================
function y = Tanh(x)
    x = 2 * x / 3;
    y = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));
    y = 1.7159 * y;
end
