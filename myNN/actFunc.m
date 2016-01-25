%==========================================================================
%Description:
%       activation function
%Input:
%       x      : data
%       type   : activation function type
%Output:
%       nn     : the neural networks after trainning
%==========================================================================
function y = actFunc(x, type)
    switch type
        case 'Sigmoid'
            y = 1 ./ (1 + exp(-x));
        case 'Tanh'
            x = 2 * x / 3;
            y = 1.7159 * tanh(x);
        case 'ReLU'
            y = max(0, x);
    end       
end