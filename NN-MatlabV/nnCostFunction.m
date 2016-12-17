function [J,Theta1,Theta2] = nnCostFunction(num_labels,X,Y,lambda,eta,Theta1,Theta2,y)

m = size(X, 1);
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
a1 = [ones(m, 1) X];
z2 = double(a1) * double(Theta1');
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
hThetaX = a3;

J = 1/m * sum(sum(-1 * Y .* log(hThetaX)-(1-Y) .* log(1-hThetaX))) + (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));
for t = 1:m,
 a1 = [1; X(t,:)'];
 z2 =  double(Theta1) *  double(a1);
 a2 = sigmoid(z2);
 a2 = [1; a2];
 z3 =  double(Theta2) *  double(a2);
 a3 = sigmoid(z3);

 Y = ([1:num_labels]==y(t))';
 delta3 = a3 - Y;
 delta2 = (Theta2'*delta3).*[1;sigmoidGradient(z2)];
 delta2 = delta2(2:end);
 
Theta1_grad =  Theta1_grad +  double(delta2)*double(a1');
Theta2_grad =  Theta2_grad +  double(delta3)*double(a2');
end

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
Theta1 = Theta1 - eta*Theta1_grad;
Theta2 = Theta2 - eta*Theta2_grad;
end
