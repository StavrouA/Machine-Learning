function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
  %theta_temp = theta;
  %theta_temp(1) = 0;  %not to regularize
  
  %J = (sum((X*theta-y).^2))/(2*m) + (lambda/(2*m))*sum(theta_temp.^2); % cost function
  
  %grad1 = ((X*theta-y)'*X(:,1))/m;                                % gradient1
  %grad2 = ((X*theta-y)'*X(:,2))/m + (lambda/m)*theta(2,1);        % gradient2
  %grad = [grad1;grad2];                                           % gradient

%test
 
h = X*theta;

J = ((h-y)' * (h-y))/(2*m) + (lambda/(2*m)) * (theta(2:length(theta)))' * theta(2:length(theta));

theta_temp = theta;
theta_temp(1) = 0;      

grad = ((1/m) * (h-y)'*X) + (lambda/m) * theta_temp';

% =========================================================================

grad = grad(:);

end
