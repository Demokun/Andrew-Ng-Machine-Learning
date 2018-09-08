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

%START CODE---------------------------------------------------------------------

%part 1 �������-----------------------------------------------------------------

%��ע����ǰ�򴫲��ļ�������У���Ҫ����bias��Ԫ����֤���躯���еĳ�����

%��ʼ��y����������������У���������ʮ������㣬������Ϊһ��Ԫ��ֵΪ0��1��5000x10�ľ���
%        ��������ǩ��y��һ��5000x1�ľ�����Ԫ��ֵΪ0-9
%        �����Ҫ��yת����һ��5000x10�ľ��������������ֵ
yNum = zeros(m, num_labels);
for i = 1:m
  yNum(i, y(i)) = 1;
end
%Ϊѵ����X��5000x400����������һ��ֵΪ1���У�5000x401��
X = [ones(size(X,1),1) X];

%��һ��ļ���
z2 = X * Theta1' ;
a2 = sigmoid(z2);

%Ϊ��һ�ε����루5000x25����������һ��ֵΪ1���У�5000x26��
a2 = [ones(size(a2,1),1) a2];

%�ڶ���ļ���
z3 = a2 * Theta2' ;
a3 = sigmoid(z3);

%�������
h = a3;
%J = 1 / m * sum ( -yNum * log ( h' ) - ( 1 - yNum ) * log ( 1 - h' ) , 2 );
for i = 1:m
  J = J + (-yNum(i, :) * log(h(i, :))' - (1.- yNum(i, :)) * log(1.- h(i, :))');
end
J = J / m;

%��������Ĵ��ۺ���
%�Ӳ��������еĵڶ��п�ʼ�����������һ��Ϊ����1��

%��ע����������ļ����������Ҫ�Ƴ�bias��Ԫ 

t1 = Theta1(:, 2:end);
t2 = Theta2(:, 2:end);
reg = (lambda / (2 * m)) * (sum(sum(t1 .^ 2)) + sum(sum(t2 .^ 2)));
J = J + reg;

%part 2 �����ݶ�-----------------------------------------------------------------
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
for t = 1:m
  a1 = X(t,:);
  %��һ��ļ���
  z2 = a1 * Theta1' ;
  a2 = sigmoid(z2);

  %Ϊ��һ�ε����루5000x25����������һ��ֵΪ1���У�5000x26��
  a2 = [ones(size(a2,1),1) a2];

  %�ڶ���ļ���
  z3 = a2 * Theta2' ;
  a3 = sigmoid(z3);
  err3 = zeros(num_labels,1);
   for k = 1:num_labels     
      err3(k) = a3(k) - (y(t) == k);
   end
  %delta3 = a3 - yNum(t,:);
  err2 = Theta2' * err3;
  err2 = err2(2:end) .* sigmoidGradient(z2');
  delta2 = delta2 + (err3 * a2);
  delta1 = delta1 + (err2 * a1);
endfor
  Theta1_temp = [zeros(size(Theta1,1),1) Theta1(:,2:end)];
  Theta2_temp = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
  Theta1_grad = 1 / m * delta1 + lambda/m * Theta1_temp;
  Theta2_grad = 1 / m * delta2 + lambda/m * Theta2_temp ;

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
