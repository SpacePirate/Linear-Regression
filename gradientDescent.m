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
    
    %fprintf('theta_0 = %f...\n', theta(1));
    %fprintf('J = %f...\n', computeCost(X, y, theta));
    
    theta_0 = theta(1);
    theta_1 = theta(2);
    
    sdiff_0 = 0;
    sdiff_1 = 0;
    for i = 1:m
      h = 0;
      for j = 1:length(theta)
        h = h + (theta(j) * X(i, j));
      end
      sdiff_0 = sdiff_0 + ((h - y(i)))*X(i, 1);
      sdiff_1 = sdiff_1 + ((h - y(i)))*X(i, 2);
    end
    
    theta_0 = theta_0 - (alpha*(1/m)*sdiff_0);
    theta_1 = theta_1 - (alpha*(1/m)*sdiff_1);
    
    theta(1) = theta_0;
    theta(2) = theta_1;
    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);



end

end
