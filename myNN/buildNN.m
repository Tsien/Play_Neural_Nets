%==========================================================================
%Description:
%      set up a neural network according to the parameter.
%Input:
%      archit: Architecture, a rwo vector indicting the number of neurons from
%      input layer to output layer 
%Output:
%      acc : the overall accuracy
%      W   : weights
%Usage:
%       NN = buildNN([784, 300, 10]) build a neural network with 784 input
%       neurons, 300 hidden neurons and 10 output neurons
%==========================================================================
function nn = buildNN(archit)
    nn.architecture = archit;
    nn.layerNum = numel(archit); % Ex(h): the number of layers
    nn.learnRate = 2; % learning rate
    
    nn.lambda = 0; % Ex(e): weight decay, L2 regularization, w = w - alpha (dEdW + lambda w)
    nn.gamma = 0; % Ex(f): Momentum
    nn.activeFunc = 'Sigmoid'; % Ex(g): the activation function of hidden neurons, Sigmoid, Tanh or ReLU
    nn.output = 'Softmax'; % Ex(g): the activation function of output neurons, Sigmoid or Softmax    

    %initial nets' weights
    for i = 2 : nn.layerNum
        nn.weights{i - 1} = rand(archit(i - 1) + 1, archit(i)) - 0.5;
        %NAG: for accelerating gradient descent that accumulates a velocity
        %vector in directions of persistent reduction in the objective across iterations. 
        nn.vW{i - 1} = zeros(size(nn.weights{i - 1}));
    end
end