function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

a = sigmoid(X*theta);
b = -y.*log(a);
c= (1-y).*(log(1-a));
d = b - c;
J = sum(d)/m;

e = (a - y);
for (i = 1: size(grad,1))
	grad(i,:) = (sum(e.*X(:,i)))/m;
end

%grad(1,1) = sum(e.*X(:,1));
%grad(2,1) = sum(e.*X(:,2));
%grad(3,1) = sum(e.*X(:,3));
%grad

% =============================================================

end
