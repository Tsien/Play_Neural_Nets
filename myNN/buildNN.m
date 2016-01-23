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
function NN = buildNN(archit)
    NN.layerNum = numel(archit); % Ex(h): the number of layers
    NN.learnRate = 0.1; % learning rate
    NN.scalLearnRate = 1; % scaling learning rate between every epoch
    NN.test = 0; % distinct training and testing
    
    NN.weightDecay = 0; % Ex(e): L2 regularization
    NN.momentum = 0.5; % Ex(f): Momentum
    NN.activeFunc = 'sigmoid'; % Ex(g): the activation function of hidden neurons
    NN.output = 'softmax'; % Ex(g): the activation function of output neurons    

end