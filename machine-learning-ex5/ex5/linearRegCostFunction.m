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

%Compute cost function
J = (X*theta - y)'*(X*theta-y) / (2*m);

%Add Regularization 
%Set theta(1) to 0 and
regTheta = theta;
regTheta(1) = 0;
reg_term = (lambda / (2*m))*sum((regTheta'*regTheta));
%Add term to cost function
J = J + reg_term;

%Do gradient calculation with regularization factor
grad = (X'*(X*theta-y))/m;
newTheta = theta;
newTheta(1,1) = 0;
grad_reg_term = (lambda*newTheta)/m;
grad = grad + grad_reg_term;





% =========================================================================

grad = grad(:);

end
