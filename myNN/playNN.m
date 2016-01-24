%==========================================================================
%Description:
%       main function for playing neural networks
%Input:
%       dataFile: contains MNIST data, images are already rescaled to [0,1] double, 60000X784
%Usage:
%       playNN('readyMNIST')
%==========================================================================
function playNN(dataFile)
    tic;
    load(dataFile);
    toc
    train_x = double(train_x') / 255;
    test_x  = double(test_x')  / 255;
    train_y = double(train_y);
    test_y  = double(test_y);
    %train_y = expLabel(train_y);%60000 X 10
    %test_y = expLabel(test_y);%10000 X 10
    % normalize by zscoring
    [train_x, mu, sigma] = zscore(train_x');%60000 X 784
    sigma = max(sigma, eps);%avoid zero sigma, -->NAN
    test_x = normalize(test_x', mu, sigma);%10000 X 784
    
    %Ex(d) Classification on MNIST database using vanilla neural networks
    rand('state', 0); %Setting the generator to the same fixed state allows computations to be repeated. 
    %set up the parameters of experiments
    exp.numEpochs = 1; 
    exp.batchSize = 100;
    nn = buildNN([784, 100, 10]);
    nn = trainNN(nn, train_x, train_y, exp);
    acc = testNN(nn, test_x, test_y);
    assert(1 - acc < 0.08, ['Too big error' num2str(1 - acc)]);

    %Ex(e) Experiment with Regularization
    
    %Ex(f) Experiment with Momentum
    
    %Ex(g) Experiment with different Activations
    
    %Ex(h) Experiemnt with different Network Topology
end

