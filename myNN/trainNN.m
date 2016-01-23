%==========================================================================
%Description:
%       Train a neural networks.
%Input:
%       nn     : neural networks including its parameters and structure
%       train_x: images, already rescale to [0,1] double, 60000X784
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
            batch_x = train_x(:, index((j - 1) * exp.batchSize + 1:j * exp.batchSize, :));
            batch_y = train_y(index((j - 1) * exp.batchSize + 1:j * exp.batchSize, :));
            %forward
                nn = forwardNN(nn, batch_x);
                nn = backProNN(nn, batch_y);
                nn = updateWNN(nn);
            %BP
            
            %Update weights
            
        end
        t = toc;
    end
end

