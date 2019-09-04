function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

positiveExamples = X(find(y==1),:);
negativeExamples = X(find(y==0),:);
plot(positiveExamples(:,1),positiveExamples(:,2),'k+',"markersize", 10);
plot(negativeExamples(:,1),negativeExamples(:,2),'ro',"markersize", 10);
%xLabel('Exam 1 score');
%yLavel('Exam 2 score');
%legend("Admitted","Not Admitted");


% =========================================================================



hold off;

end
