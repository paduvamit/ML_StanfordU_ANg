function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.

%Recode y from 5000x1 to 10x5000
y_expanded = zeros(num_labels, m);
for i = 1:m,
    y_expanded(y(i),i) = 1;
end;

a1 = [ ones(1,m) ; X']; % add row for biases

z2 = Theta1 * a1;
a2 = [ ones(1,m) ; sigmoid(z2)]; % add row for biases

z3 = Theta2 * a2;
predicted = sigmoid(z3); % no need to add row for biases because this is the output layer .i.e., a3

sum_over_all_output_units = sum( ( y_expanded .* log(predicted) ) + ( (1 - y_expanded) .* log(1 - predicted) ) );

sum_over_all_examples = sum(sum_over_all_output_units);

J = (-1/m) * sum_over_all_examples;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

d_3 = predicted - y_expanded;
d_2 = (Theta2(:,2:end)' * d_3) .* sigmoidGradient(z2);

Theta1_grad = (1/m) * d_2 * a1';
Theta2_grad = (1/m) * d_3 * a2';

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

J_reg = ( lambda/(2*m) ) * ( sum( (Theta1.^2)(:,2:end)(:) ) + sum( (Theta2.^2)(:,2:end)(:) ) );

Theta1_reg = (lambda/m) * (Theta1)(:,2:end);
Theta2_reg = (lambda/m) * (Theta2)(:,2:end);

% The bias gradients should be spared regularization
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + Theta1_reg;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + Theta2_reg;

J = J + J_reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
