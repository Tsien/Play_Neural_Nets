%==========================================================================
%Description:
%       main function for playing neural networks
%Input:
%       dataFile: contains MNIST data, images are already rescaled to [0,1] double, 60000X784
%Usage:
%       playNN('readyMNIST')
%==========================================================================
function playNN(dataFile)
%==========================================================================
    %Data pre-process
    load(dataFile);
    train_y = expLabel(train_y);%60000 X 10
    test_y = expLabel(test_y);%10000 X 10
    % normalize by zscoring
    train_x = zscore(train_x', 0, 2);%60000 X 784
    test_x = zscore(test_x', 0, 2);%10000 X 784
    % build neural networks
    rand('state', 0); %Setting the generator to the same fixed state allows computations to be repeated.
    nn = buildNN([784, 100, 10]);
% %==========================================================================
%     %Ex(d) Classification on MNIST database using vanilla neural networks 
%     %set up the parameters of experiments
%     exp.numEpochs = 20; 
%     exp.batchSize = 100;
%     exp.valid = 1; % cross-validation to decide the stopping criteria
%     nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
%     plot([1:exp.numEpochs],nn.trainAcc,[1:exp.numEpochs],nn.validAcc, [1:exp.numEpochs], nn.testAcc);
%     legend('Training set', 'Validation Set', 'Test Set');
%     %check numerical gradient 
%     %nn = compGrad(nn, train_x(1:100, :), train_y(1:100, :));
%     acc = testNN(nn, test_x, test_y);
% %==========================================================================
%     %Ex(e) Experiment with Regularization(L2 w = w - alpha (dEdW + lambda w))
%     nn.lambda = 0.0001;
%     exp.numEpochs = 20;
%     exp.batchSize = 100;
%     exp.valid = 0;
%     nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
%     plot([1:exp.numEpochs],nn.trainAcc, [1:exp.numEpochs], nn.testAcc);
%     legend('Training set', 'Test Set');
%     acc = testNN(nn, test_x, test_y);
% %==========================================================================    
%     %Ex(f) Experiment with Momentum
%     nn.gamma = 0.9;
%     nn.lambda = 0.0001;
%     exp.numEpochs = 20;
%     exp.batchSize = 100;
%     exp.valid = 0;
%     nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
%     plot([1:exp.numEpochs],nn.trainAcc, [1:exp.numEpochs], nn.testAcc);
%     legend('Training set', 'Test Set');
%     acc = testNN(nn, test_x, test_y);
% %==========================================================================
%     %Ex(g) Experiment with different Activations--Funny Tanh
%     nn.activeFunc = 'Tanh';
%     exp.numEpochs = 20;
%     exp.batchSize = 100;
%     exp.valid = 0;
%     nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
%     plot([1:exp.numEpochs],nn.trainAcc, [1:exp.numEpochs], nn.testAcc);
%     legend('Training set', 'Test Set');
%     acc = testNN(nn, test_x, test_y);
% %==========================================================================
%     %Ex(g) Experiment with different Activations--ReLU
%     nn.activeFunc = 'ReLU';
%     exp.numEpochs = 100;
%     exp.batchSize = 100;
%     exp.valid = 0;
%     nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
%     plot([1:exp.numEpochs],nn.trainAcc, [1:exp.numEpochs], nn.testAcc);
%     legend('Training set', 'Test Set');
%     acc = testNN(nn, test_x, test_y);
% %==========================================================================
%     %Ex(h) Experiemnt with different Network Topology, 50 hidden 
%     nn = buildNN([784, 50, 10]);
%     exp.numEpochs = 20; 
%     exp.batchSize = 100;
%     exp.valid = 1; % cross-validation to decide the stopping criteria
%     nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
%     plot([1:exp.numEpochs],nn.trainAcc,[1:exp.numEpochs],nn.validAcc, [1:exp.numEpochs], nn.testAcc);
%     legend('Training set', 'Validation Set', 'Test Set');
%     acc = testNN(nn, test_x, test_y);
% %==========================================================================
%     %Ex(h) Experiemnt with different Network Topology, 200 hidden
%     nn = buildNN([784, 200, 10]);
%     exp.numEpochs = 20; 
%     exp.batchSize = 100;
%     exp.valid = 1; % cross-validation to decide the stopping criteria
%     nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
%     plot([1:exp.numEpochs],nn.trainAcc,[1:exp.numEpochs],nn.validAcc, [1:exp.numEpochs], nn.testAcc);
%     legend('Training set', 'Validation Set', 'Test Set');
%     acc = testNN(nn, test_x, test_y);
%==========================================================================
    %Ex(h) Experiemnt with different Network Topology, 2 hidden layer
    nn = buildNN([784, 100, 100, 10]);
    exp.numEpochs = 20; 
    exp.batchSize = 100;
    exp.valid = 1; % cross-validation to decide the stopping criteria
    nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
    plot([1:exp.numEpochs],nn.trainAcc,[1:exp.numEpochs],nn.validAcc, [1:exp.numEpochs], nn.testAcc);
    legend('Training set', 'Validation Set', 'Test Set');
    acc = testNN(nn, test_x, test_y);
end

