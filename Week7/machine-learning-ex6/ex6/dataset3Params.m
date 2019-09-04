function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

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

%values_to_try = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
%lowest_cv_error = 1;

%for c_index = 1:length(values_to_try),
%    C_value = values_to_try(c_index);
%    for s_index = 1:length(values_to_try),
%        sigma_value = values_to_try(s_index);
%        model= svmTrain(X, y, C_value, @(x1, x2) gaussianKernel(x1, x2, sigma_value));
%        predictions = svmPredict(model, Xval);
%        cv_error = mean(double(predictions ~= yval));
%        if(cv_error < lowest_cv_error),
%            lowest_cv_error = cv_error;
%            C = C_value;
%            sigma = sigma_value;
%        end;
%    end;
%end;

% =========================================================================

end
