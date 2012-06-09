function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = size(X,2);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	tempTheta = zeros(size(theta));
	for(p = 1: n)
		pp = 0;
		for(i = 1 : m)
			a = 0;	
			for(j = 1 : n)
				a = a + theta(j,1) * X(i,j);
			end
			k = (a - y(i,1))*X(i, p);
			pp = pp+k;		
		end
		#pp
		pp = (pp*alpha)/m
		tempTheta(p,1) = theta(p,1) - pp;	
	end
	
	theta = tempTheta; 
#	a = 0;
#	for (i = 1 : m)
#		a = a + ((theta(1,1)*X(i,1) + theta(2,1)*X(i,2)) - y(i,1))*X(i,1); 
#	end
#	a = (a*alpha)/m
#	temp1 = theta(1,1) - a;

#	b = 0;
#	for (i = 1 : m)
#		b = b + ((theta(1,1)*X(i,1) + theta(2,1)*X(i,2)) - y(i,1))*X(i,2); 
#	end
#	b = (b*alpha)/m
#	temp2 = theta(2,1) - b;

#	theta(1,1) = temp1;
#	theta(2,1) = temp2;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
