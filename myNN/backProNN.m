%==========================================================================
%Description:
%       backpropagation, calculate the delta and dE/dW
%Input:
%       nn     : neural networks including its parameters and structure
%Output:
%       nn     : the neural networks after backpropagation, containing
%       dE/dW for updating weights
%==========================================================================
function nn = backProNN(nn)
    n = nn.layerNum;
    for i = (n - 1) : -1 : 2
        % Derivative of the activation function
        switch nn.activeFunc 
            case 'Sigmoid'
                deri_actFuc = nn.activation{i} .* (1 - nn.activation{i});
            case 'Tanh'
                deri_actFuc = 1 - nn.activation{i}.^2;
            case 'ReLU'%f(z) = max(0, z): 1 when z>0, 0 when z<=0
                deri_actFuc = zeros(size(nn.activation{i}));
                deri_actFuc(find(nn.activation{i} > 0)) = 1;
        end
        
        if i+1==n % no bias term to be removed             
            nn.delta{i} = (nn.delta{i + 1} * nn.W{i}) .* deri_actFuc;
            nn.dEdW{i} = (nn.delta{i + 1}' * nn.activation{i}) / size(nn.delta{i + 1}, 1);
        else % the bias term needs to be removed
            nn.delta{i} = (nn.delta{i + 1}(:, 2:end) * nn.W{i}) .* deri_actFuc;
            nn.dEdW{i} = (nn.delta{i + 1}(:, 2:end)' * nn.activation{i}) / size(nn.delta{i + 1}, 1);
        end
    end    
end