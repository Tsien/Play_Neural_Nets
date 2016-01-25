%==========================================================================
%Description:
%       Compare the gradient using numerical differences with the one using backpropagation
%Input:
%       nn     : neural networks including its parameters and structure
%       x      : input of neural nets, m X 784, a small subset of data
%       y      : labels
%Output:
%       nn     : neural networks including statistics on the differences
%==========================================================================
function nn = compGrad(nn, x, y)
    epsilon = 1e-5;
    n = nn.layerNum;
    nn.checkMax = -1;% max difference
    nn.checkMean = 0;% mean difference
    num = 0;
    for l = 1 : n - 1
        m = size(nn.weights{l});
        num = num + numel(nn.weights{l});
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
                delta = abs(dEdW - nn.dEdW{l}(j, i));
                nn.checkMean = nn.checkMean + delta;
                nn.checkMax = max(delta, nn.checkMax);
            end
        end
    end
    nn.checkMean = nn.checkMean/num;
end


