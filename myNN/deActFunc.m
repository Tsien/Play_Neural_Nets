%==========================================================================
%Description:
%       the derivate of activation function
%Input:
%       x      : data
%       type   : activation function type
%Output:
%       nn     : the neural networks after trainning
%==========================================================================
function y = deActFunc(x, type)
    switch type
        case 'Sigmoid'
            y = x .* (1 - x);
        case 'Tanh'
            y = 1.7159 * (1 - ( x / 1.7159).^2) * 2 /3;
        case 'ReLU'%f(z) = max(0, z): 1 when z>0, 0 when z<=0
            y = double(x > 0);
    end
end