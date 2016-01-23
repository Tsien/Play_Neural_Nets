%==========================================================================
%Description:
%       main function for playing neural networks
%Input:
%       dataFile: contains MNIST data, images are already rescaled to [0,1] double, 60000X784
%Usage:
%       nn = trainNN(train_x, train_y, valid_x, valid_y, options)
%==========================================================================
function playNN(dataFile)

    load(dataFile);
    % normalize by zscoring
    [train_x, mu, sigma] = zscore(train_x);
    sigma = max(sigma, eps);%avoid zero sigma, -->NAN
    test_x = normalize(test_x, mu, sigma);
    
    %Ex(d) Classification on MNIST database using vanilla neural networks
    %rand('state', 0); %Setting the generator to the same fixed state allows computations to be repeated. 
    %set up the parameters of experiments
    exp.numEpochs = 1; 
    exp.batchsize = 100;
    nn = buildNN([784, 300, 10]);
    nn = trainNN(nn, train_x, train_y, exp);
    [er, bad] = testNN();
    assert(er < 0.08, 'Too big error');

    %Ex(e) Experiment with Regularization
    
    %Ex(f) Experiment with Momentum
    
    %Ex(g) Experiment with different Activations
    
    %Ex(h) Experiemnt with different Network Topology
end

