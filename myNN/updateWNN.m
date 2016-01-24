%==========================================================================
%Description:
%       update all weights at the same time based on the learning rule
%Input:
%       nn     : neural networks containing dE/dW for updating weights
%Output:
%       nn     : the neural networks after updating weights
%==========================================================================
function nn = updateWNN(nn)
    n = nn.layerNum;
    for i = 1 : (n - 1)
        deltaW = nn.learnRate * nn.dEdW{i};            
        nn.weights{i} = nn.weights{i} - deltaW';% w = w - alpha * dE/dW
    end    
end

