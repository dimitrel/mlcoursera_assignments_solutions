function [mu, sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%


m1=mean(X);
%sigma2(:,1)=var(X);


b=zeros(n,1);
for i=1:n
    s=X(:,i)-m1(1,i);
    s=s.^2;
    b(i)=(sum(s))/m;
end

sigma2=b;
mu(:,1)=mean(X);
%sigma2=[s1,s2];
%}




% =============================================================


end
