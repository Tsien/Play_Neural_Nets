%==========================================================================
%Description:
%       expand labels to the range of [0...9]
%Input:
%       y      : labels
%Output:
%       labels : expanded labels
%==========================================================================
function labels = expLabel(y)
    num = numel(y);
    labels = zeros(num, 10);
    for i = 1 : num
        labels(i, y(i) + 1) = 1;
    end
end