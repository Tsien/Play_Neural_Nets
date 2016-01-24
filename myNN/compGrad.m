%==========================================================================
%Description:
%       Compare the gradient using numerical differences with the one using backpropagation
%Input:
%       nn     : neural networks including its parameters and structure
%       x      : input of neural nets, m X 784, a small subset of data
%       y      : labels
%==========================================================================
function compGrad(nn, x, y)
    epsilon = 1e-5;
    n = nn.layerNum;
    for l = 1 : n - 1
        m = size(nn.weights{l});
        for i = 1 : m(1)
            for j = 1 : m(2)
                tmp = nn.weights{l}(i, j);
                %calculate E(w + epsilon)
                nn.weights{l}(i, j) = tmp + epsilon;
                rand('state', 0);
                nn = forwardNN(nn, x, y);
                E1 = nn.error;
                %calculate E(w - epsilon)
                nn.weights{l}(i, j) = tmp - epsilon;
                rand('state', 0);
                nn = forwardNN(nn, x, y);
                E2 = nn.error;
                %calculate the numerical gradient approximation
                dEdW = (E1 - E2) / (2 * epsilon);
                %Compare
                delta = abs(dEdW - nn.dEdW{l}(i, j));
                assert(delta < 1e-3, 'numerical gradient checking failed');
            end
        end
    end
end


