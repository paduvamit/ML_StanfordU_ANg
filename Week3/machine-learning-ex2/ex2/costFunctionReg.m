function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

theta_1_to_n = theta(2:length(theta));

[J_without_regularization, grad_without_regularization] = costFunction(theta, X, y);

J = J_without_regularization + ((lambda/(2*m)) * theta_1_to_n' * theta_1_to_n);

grad(1) = grad_without_regularization(1);
grad(2:length(grad)) = grad_without_regularization(2:length(grad)) .+ ((lambda/m) * theta_1_to_n);

% =============================================================

end
