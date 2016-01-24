%==========================================================================
%Description:
%       forward process neural networks.
%Input:
%       nn     : neural networks including its parameters and structure
%       x      : input of neural nets, m X 784
%       y      : labels
%Output:
%       nn     : the neural networks after trainning
%==========================================================================
function nn = forwardNN(nn, x, y)
    num = size(x, 1);
    % the output of the input layer
    nn.activation{1} = [ones(num, 1), x];
    for i = 2 : nn.layerNum - 1
        tmp = nn.activation{i - 1} * nn.weights{i - 1};
        switch nn.activeFunc
            case 'Sigmoid'
                nn.activation{i} = Sigmoid(tmp);%num X 784 * 784 X num 
            case 'Tanh'
                nn.activation{i} = Tanh(tmp);%num X 784 * 784 X num
            case 'ReLU'
                nn.activation{i} = ReLU(tmp);%num X 784 * 784 X num
        end       
        %add bias
        nn.activation{i} = [ones(num, 1), nn.activation{i}];
    end
    %for the softmax output layer
    tmp = nn.activation{nn.layerNum - 1} * nn.weights{nn.layerNum - 1};
    nn.activation{nn.layerNum} = Softmax(tmp);
    
    % cross-entropy error(Xent)
    nn.error = -sum(sum(y .* log(nn.activation{nn.layerNum}))) / num; 
    % delta of the output layer.
    nn.delta{nn.layerNum} = -(y - nn.activation{nn.layerNum}); 
end
