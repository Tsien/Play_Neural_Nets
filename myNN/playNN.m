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
%==========================================================================
    %set up parameters
    %Setting the generator to the same fixed state allows computations to be repeated.
    rand('state', 0); 
    % build neural networks
    nn = buildNN([784, 100, 10]);
    exp.numEpochs = 30; 
    exp.batchSize = 100;
    exp.valid = 1; % cross-validation to decide the stopping criteria
%==========================================================================
    %Ex(d) Classification on MNIST database using vanilla neural networks 
    %set up the parameters of experiments
    nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
    plot([1:exp.numEpochs],nn.trainAcc,[1:exp.numEpochs],nn.validAcc, [1:exp.numEpochs], nn.testAcc);
    legend('Training Accuracy', 'Validation Accuracy', 'Test Accuracy');
    %check numerical gradient 
    %nn = compGrad(nn, train_x(1:100, :), train_y(1:100, :));
    acc = testNN(nn, test_x, test_y);
%==========================================================================
    %Ex(e) Experiment with Regularization(L2 w = w - alpha (dEdW + lambda w))
    nn.lambda = 0.0001;
    nn.learnRate = 0.5;
    nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
    plot([1:exp.numEpochs],nn.trainAcc,[1:exp.numEpochs],nn.validAcc, [1:exp.numEpochs], nn.testAcc);
    legend('Training Accuracy', 'Validation Accuracy', 'Test Accuracy');
    acc = testNN(nn, test_x, test_y);
%==========================================================================    
    %Ex(f) Experiment with Momentum
    exp.numEpochs = 100; 
    nn.gamma = 0.9;
    nn.learnRate = 0.25;
    nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
    plot([1:exp.numEpochs],nn.trainAcc,[1:exp.numEpochs],nn.validAcc, [1:exp.numEpochs], nn.testAcc);
    legend('Training Accuracy', 'Validation Accuracy', 'Test Accuracy');
    acc = testNN(nn, test_x, test_y);
%==========================================================================
    %Ex(g) Experiment with different Activations--Funny Tanh
    nn.activeFunc = 'Tanh';
    nn.learnRate = 0.1;
    exp.numEpochs = 100;
    nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
    plot([1:exp.numEpochs],nn.trainAcc,[1:exp.numEpochs],nn.validAcc, [1:exp.numEpochs], nn.testAcc);
    legend('Training Accuracy', 'Validation Accuracy', 'Test Accuracy');
    acc = testNN(nn, test_x, test_y);
%==========================================================================
    %Ex(g) Experiment with different Activations--ReLU
    nn.activeFunc = 'ReLU';
    nn.learnRate = 0.05;
    exp.numEpochs = 100;
    nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
    plot([1:exp.numEpochs],nn.trainAcc,[1:exp.numEpochs],nn.validAcc, [1:exp.numEpochs], nn.testAcc);
    legend('Training Accuracy', 'Validation Accuracy', 'Test Accuracy');
    acc = testNN(nn, test_x, test_y);
%==========================================================================
    %Ex(h) Experiemnt with different Network Topology, 50 hidden 
    rand('state', 0); 
    nn = buildNN([784, 50, 10]);
    nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
    plot([1:exp.numEpochs],nn.trainAcc,[1:exp.numEpochs],nn.validAcc, [1:exp.numEpochs], nn.testAcc);
    legend('Training Accuracy', 'Validation Accuracy', 'Test Accuracy');
    acc = testNN(nn, test_x, test_y);
%==========================================================================
    %Ex(h) Experiemnt with different Network Topology, 200 hidden
    rand('state', 0); 
    nn = buildNN([784, 200, 10]);
    nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
    plot([1:exp.numEpochs],nn.trainAcc,[1:exp.numEpochs],nn.validAcc, [1:exp.numEpochs], nn.testAcc);
    legend('Training Accuracy', 'Validation Accuracy', 'Test Accuracy');
    acc = testNN(nn, test_x, test_y);
%==========================================================================
    %Ex(h) Experiemnt with different Network Topology, 2 hidden layer
    rand('state', 0); 
    nn = buildNN([784, 100, 100, 10]);
    nn = trainNN(nn, train_x, train_y, exp, test_x, test_y);
    plot([1:exp.numEpochs],nn.trainAcc,[1:exp.numEpochs],nn.validAcc, [1:exp.numEpochs], nn.testAcc);
    legend('Training Accuracy', 'Validation Accuracy', 'Test Accuracy');
    acc = testNN(nn, test_x, test_y);
end

