%==========================================================================
%Description:
%       backpropagation
%Input:
%       nn     : neural networks including its parameters and structure
%       x      : input of neural nets, m X 784
%Output:
%       nn     : the neural networks after trainning
%==========================================================================
function nn = backProNN(nn)
    d{n} = - nn.e;
    for i = (n - 1) : -1 : 2
        % Derivative of the activation function
        switch nn.activeFunc 
            case 'Sigmoid'
                d_act = nn.a{i} .* (1 - nn.a{i});
            case 'Tanh'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
            case 'ReLU'
        end
        
        % Backpropagate first derivatives
        if i+1==n % in this case in d{n} there is not the bias term to be removed             
            d{i} = (d{i + 1} * nn.W{i}) .* d_act; % Bishop (5.56)
        else % in this case in d{i} the bias term has to be removed
            d{i} = (d{i + 1}(:,2:end) * nn.W{i}) .* d_act;
        end
    end

    for i = 1 : (n - 1)
        if i+1==n
            nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
        else
            nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);      
        end
    end    
end