function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

size(X)
a = sigmoid(X*theta);
b = -y.*log(a);
c= (1-y).*(log(1-a));
d = b - c;
J1 = sum(d)/m;
newTheta = theta(2:size(theta,1),1);
J2 = (lambda/(2*m)) * (sum(newTheta.*newTheta));
J = J1+J2;

difference = (a - y);

grad(1,:) = sum(difference.*X(:,1))/m ;
#for (i = 2 : size(grad,1))
#	grad(i,:) = ((sum(difference.*X(:,i)))/m) + ((lambda*theta(i,1))/m);
#end

#size(difference);
ppq = X(:,2:end);
newdiff = repmat(difference, [1 size(ppq,2)]);
#size(newdiff);
#size(ppq);
pp = sum(newdiff.*ppq)'./m;
#size(pp)
qq = (lambda.*theta(2:end,1))./m;
#size(qq)
grad(2:end,:) = pp + qq;







% =============================================================

grad = grad(:);

end
