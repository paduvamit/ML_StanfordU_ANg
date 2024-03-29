function p = predictLinear(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICTLINEAR(Theta1, Theta2, X) outputs the predicted value of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = tanh([ones(m, 1) X] * Theta1');
h2 = [ones(m, 1) h1] * Theta2';
p = h2;

% =========================================================================


end
