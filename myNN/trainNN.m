%==========================================================================
%Description:
%       Train a neural networks.
%Input:
%       nn     : neural networks including its parameters and structure
%       train_x: training set, images, already rescale to [0,1] double, 60000X784
%       train_y: training set, labels
%       exp    : other parameters such as number of epoches, batchsize etc.
%       test_x : test set, images, already rescale to [0,1] double, 60000X784
%       test_y : test set, labels
%Output:
%       nn     : the neural networks after trainning
%==========================================================================
function nn = trainNN(nn, train_x, train_y, exp, test_x, test_y)
    numSample = size(train_x, 1);
    if exp.valid == 1
        pos = floor(numSample/6);
        index = randperm(numSample); % randomly select samples
        valid_x = train_x(index(1:pos), :);
        valid_y = train_y(index(1:pos), :);
        train_x = train_x(index(pos + 1:end), :);
        train_y = train_y(index(pos + 1:end), :);
        numSample = numSample - pos;
    end
    numBatches = numSample / exp.batchSize;
    for i = 1 : exp.numEpochs
        tic;
        index = randperm(numSample); % randomly select samples
    	for j = 1 : numBatches
            batch_x = train_x(index((j - 1) * exp.batchSize + 1:j ...
                * exp.batchSize), :);
            batch_y = train_y(index((j - 1) * exp.batchSize + 1:j ...
                * exp.batchSize), :);
            %forward
            nn = forwardNN(nn, batch_x, batch_y);
            %Backpropagation to update weights
            nn = backProNN(nn);
        end
        toc
        %plot training accuracy vs. number of training iterations
        nn.trainAcc(i) = testNN(nn, train_x, train_y);
        nn.testAcc(i) = testNN(nn, test_x, test_y);
        if exp.valid == 1
            nn.validAcc(i) = testNN(nn, valid_x, valid_y);
        end
    end
end

