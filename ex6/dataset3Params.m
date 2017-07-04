function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
values=[0.01,0.03,0.1,0.3,1,3,10,30];
error=zeros((length(values)^2),3);
k=1;
for i=1:length(values)
    for j=1:length(values)
        C=values(i);
        sigma=values(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        error(k,1)= mean(double(predictions ~= yval));
        error(k,2)=C;
        error(k,3)=sigma;
        k=k+1;
    end
end

min_err=find(error(:,1)==min(error(:,1)));
min_err=min_err(1);
C=error(min_err,2);
sigma=error(min_err,3);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%predictions = svmPredict(model, Xval)
%mean(double(predictions ~= yval))




% =========================================================================

end
