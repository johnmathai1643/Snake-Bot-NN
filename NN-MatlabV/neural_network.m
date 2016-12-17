%% Initialization

clear ; close all; clc;
%load('ORLFACEDATABASE.mat')
TRAIN = load('TRAIN_ULTI.mat');
TEST = load('TEST.mat');
TRAIN = TRAIN.TRAIN;
TEST = TEST.TEST
%y_train_test;
%C = C';
y_TRAIN = load('y_TRAIN_ULTI.mat');
y = y_TRAIN.y_TRAIN;
Y_TEST = load('y_TEST.mat');
Y_TEST = Y_TEST.y_TEST;

PCA;
TRAIN = mat_transform;
input_layer_size  = 1845;
hidden_layer_size = 64;
num_labels = 2;   
                          
%y = Y_TRAIN;
%TRAIN = C;
lambda = 0; % 5
eta = 0.75; %0.25
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

m = size(TRAIN, 1);
Y = zeros(m,num_labels);
for i = 1:m
    Y(i,y(i)) = 1;
end


for i = 1:1:300 %100
   [J,Theta1,Theta2] = nnCostFunction(num_labels, TRAIN, Y, lambda, eta,initial_Theta1,initial_Theta2,y);
   initial_Theta1 = Theta1;initial_Theta2=Theta2;
   fprintf('Iteration number:%d :: Cost Function(J):%d\n',i,J);
end
                          
pred = predict(Theta1, Theta2, TRAIN);
TEST = double(TEST);
TEST = 1.0*TEST/255;
u_rep_test = double(repmat(u,size(TEST,1),1));
TEST = double(TEST);
TEST = TEST - u_rep_test;
sig = std(TEST);
sig_rep = double(repmat(sig,size(TEST,1),1));
TEST = TEST./sig_rep;

test_transform = double(TEST)*double(eig_v);
test_transform = double(test_transform);
u_test = mean(test_transform);
sig = std(test_transform);
u_rep_test = double(repmat(u_test,size(test_transform,1),1));
sig_rep_test = double(repmat(sig,size(test_transform,1),1));
test_transform = test_transform - u_rep_test;
test_transform = test_transform./double(sig_rep_test);

pred2 = predict(Theta1, Theta2, test_transform);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred2 == Y_TEST)) * 100);
