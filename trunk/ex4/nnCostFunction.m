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
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

#p1 = size(Theta1_grad)
#p2 =size(Theta2_grad)
#m
#input_layer_size
#hidden_layer_size
#num_labels

#sizey = size(y)

#a1 = [ones(size(X,1),1) X];

#z2 = Theta1*a1';
#a2 = sigmoid(z2);

#a2 = a2';
#a2 = [ones(size(a2,1),1) a2];

#z3 = Theta2*a2';
#h = sigmoid(z3);

#h = h';
#size(h') ; #5000 X 10

#size(Theta1)
#size(Theta2)

total = 0; 
Del2 = 0;
Del1 = 0;

for (i = 1 : m)

	a1 = X(i,:);
	a1 = [ones(size(a1,1),1) a1];  # 1 X 401
	#size(a1)
	z2 = Theta1*a1';
	a2 = sigmoid(z2); # 1 X 25	

	a2 = a2'; # 25 X 1
	#size(a2)

	a2 = [ones(size(a2,1),1) a2]; # 26 X 1
	#size(a2)	
	z3 = Theta2*a2';
	a3 = sigmoid(z3); # 10 x 1	

	h = a3'; # 1 X 10
	subTotal = 0;
	[val, index] = max(h);
	currentY = zeros(num_labels, 1); # 10 X 1
	currentY(y(i,1), 1) = 1;

	#backprop
	del_3 = a3 - currentY; # 10 X 1	
	del_2 = (Theta2'*del_3); # (26 x 10) X (10 X 1) = 26 X 1
	del_2 = del_2.*(sigmoidGradient([1; z2])); # 26 X 1
	del_2 = del_2(2:end); # 25 X 1

	Del1 = Del1 + (del_2 * a1); # (25 X 1) * (1 X 401) = 25 X 401
	#size(Del1)
	Del2 = Del2 + (del_3 * a2); #  (10 X 1) * (1 X 26) = 10 X 26	
	#size(Del2)

	#cost calculation
	for (k = 1 : num_labels)
		a = -currentY(k,1)*log(h(1,k));
		b = (1-currentY(k,1))*log(1-h(1,k));
		c = a - b;
		subTotal = subTotal + c;
	end
	total = total + subTotal;
end

J = total/m;
Theta1_grad = Del1./m;
Theta2_grad = Del2./m;

Theta1_grad(:, 2:end) = Del1(:, 2:end)./m + lambda/m .* Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Del2(:, 2:end)./m + lambda/m .* Theta2(:, 2:end);

#size(Theta1)
#size(Theta2)

thetaSum1 = 0;
temp = 0;
for (j = 1 : size(Theta1,1))
	temp = 0;
	for(k = 2 : size(Theta1,2))
		temp = temp + Theta1(j,k)^2;
	end
	thetaSum1 = thetaSum1 + temp;
end

thetaSum2 = 0;
for (j = 1 : size(Theta2,1))
	temp = 0;
	for(k = 2 : size(Theta2,2))
		temp = temp + Theta2(j,k)^2;
	end
	thetaSum2 = thetaSum2 + temp;
end

J = J + (lambda/(2*m))*(thetaSum1 + thetaSum2);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
