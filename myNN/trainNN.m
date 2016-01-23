%==========================================================================
%Description:
%       Train a neural networks.
%Input:
%       nn     : neural networks including its parameters and structure
%       train_x: images, already rescale to [0,1] double, 784X60000
%       train_y: labels
%       valid_x: images used as validate set
%       valid_y: labels
%       exp    : other parameters such as number of epoches, batchsize etc.
%Output:
%       nn     : the neural networks after trainning
%==========================================================================
function nn = trainNN(nn, train_x, train_y, exp, valid_x, valid_y)
    numSample = size(train_x, 1);
    numBatches = numSample / exp.batchSize;
    for i = 1 : exp.numEpochs
        tic;
        index = randperm(numSample); % randomly select samples
	for j = 1 : numBatches
        end
        t = toc;
    end
end
