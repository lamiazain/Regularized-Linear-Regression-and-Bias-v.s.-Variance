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
%X=[ones(size(X)) X]
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h_theta=X*theta ;%mx2  2x1 =mx1;

Reg_term=(lambda/(2*m))*sum(theta(2:end).^2);
J=((sum((h_theta-y).^2))/(2*m))+ Reg_term;

%grady(1)=sum(X(:,1)'*(h_theta-y))/m;
%grady(2:end)=(sum(X(:,2:end)'*(h_theta-y))/m)+((lambda/m).*theta(2:end));


grad(1)=X(:,1)'*(h_theta-y)/m;
grad(2:end)=(X(:,2:end)'*(h_theta-y))/m+((lambda/m).*theta(2:end));



% =========================================================================

grad = grad(:);
%grady = grady(:)

end
