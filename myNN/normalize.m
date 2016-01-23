%==========================================================================
%Description:
%       normalize data using specific mean and standard deviation
%Input:
%       data   : original data
%       mu     : mean
%       sigma  : standard deviation  
%Output:
%       data   : normalized data
%==========================================================================
function data = normalize(data, mu, sigma)
    data = bsxfun(@minus, data, mu);
    data = bsxfun(@rdivide, data, sigma);
end

