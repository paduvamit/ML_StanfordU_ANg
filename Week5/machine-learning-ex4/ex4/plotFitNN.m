function plotFitNN(min_x, max_x, Theta1, Theta2)
hold on;

% We plot a range slightly bigger than the min and max values to get
% an idea of how the fit will vary outside the range of the data points
x = (min_x - 5: 0.1 : max_x + 5)';

%x = [ones(size(x, 1), 1) x];

% Plot
plot(x, predictLinear(Theta1, Theta2, x), '--', 'LineWidth', 2)

% Hold off to the current figure
hold off

end
