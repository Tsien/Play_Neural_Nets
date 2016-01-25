%==========================================================================
%Description:
%       backpropagation, calculate the delta and dE/dW
%Input:
%       nn     : neural networks including its parameters and structure
%Output:
%       nn     : the neural networks after backpropagation after updating weights
%==========================================================================
function nn = backProNN(nn)
    n = nn.layerNum;
    num = size(nn.delta{n}, 1);
    for i = (n - 1) : -1 : 1
        % Derivative of the activation function
        switch nn.activeFunc 
            case 'Sigmoid'
                deri_actFuc = nn.activation{i} .* (1 - nn.activation{i});
            case 'Tanh'
                deri_actFuc = 1.7159 * (1 - (nn.activation{i} / 1.7159).^2) * 2 /3;
            case 'ReLU'%f(z) = max(0, z): 1 when z>0, 0 when z<=0
                deri_actFuc = zeros(size(nn.activation{i}));
                deri_actFuc(find(nn.activation{i} > 0)) = 1;
        end
        
        %Nesterov’s Accelerated Gradient
        nn.weights{i} = nn.weights{i} + nn.gamma * nn.vW{i};
        
        if i+1==n % no bias term to be removed
            %calculate delta of every layer
            nn.delta{i} = (nn.delta{i + 1} * nn.weights{i}') .* deri_actFuc;%(100 X 10) X (300 X 10)' .X (100 X 300)
            %calculate the derivate of every layer
            nn.dEdW{i} = (nn.delta{i + 1}' * nn.activation{i}) / num; %(100 X 300)' X (100 X 10)
        else % the bias term needs to be removed
            nn.delta{i} = (nn.delta{i + 1}(:, 2:end) * nn.weights{i}') .* deri_actFuc;
            nn.dEdW{i} = (nn.delta{i + 1}(:, 2:end)' * nn.activation{i}) / num;
        end
        %weight decay, L2 regularization, w = w - alpha (dEdW + lambda w)
        if nn.lambda > 0
           nn.dEdW{i} = nn.dEdW{i} + nn.lambda * [zeros(size(nn.weights{i}, 2), 1), nn.weights{i}(2:end, :)'];
        end
    end

    %update all weights at the same time based on the learning rule
    for i = 1 : (n - 1)
        deltaW = nn.learnRate * nn.dEdW{i};   
        %Nesterov’s Accelerated Gradient
        nn.vW{i} = nn.gamma * nn.vW{i} + deltaW';
        deltaW = nn.vW{i};
        %Update weights
        nn.weights{i} = nn.weights{i} - deltaW;% w = w - alpha * dE/dW
    end    
end