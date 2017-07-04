function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
b=zeros(size(X, 2),1);


    for i=1:m %training sets
        b=b+((X(i,:)*theta-y(i))*X(i,:)).'; % !! Vectorized---hypothesis-gradient
        
          % Solution before----b(1)=b(1)+(X(i,:)*theta-y(i))*X(i,1);
          
    end
   
    %Alternative!--Works
    %b(1)=sum((X(:,:)*theta-y(:)).*X(:,1)); %X(i)-->training set, X(,j)-->feature
    %b(2)=sum((X(:,:)*theta-y(:)).*X(:,2));
    
    theta=theta-(alpha/m)*b;
    %theta(2)=theta(2)-(alpha/m)*b(2);


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
