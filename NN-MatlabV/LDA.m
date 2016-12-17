load TRAINTEST2D.mat
mat = [TRAIN{6}{1},TRAIN{6}{2},TRAIN{6}{3},TRAIN{6}{4}];
mat = mat';
u = mean(mat);
u_rep = repmat(u,size(mat,1),1);
S = (mat - u_rep)'*(mat - u_rep);
Covariance = S/51;
[V,D] = eig(S);
eig_v = V(:,1);
mat_transform = mat*eig_v;

subplot(1,2,1);
title('PCA for Training Data');
hold on
plot(mat_transform(1:13,:),zeros(1,13),strcat('r','*'));
hold on
plot(mat_transform(14:26,:),zeros(1,13),strcat('g','o'));
hold on
plot(mat_transform(27:39,:),zeros(1,13),strcat('b','+'));
hold on
plot(mat_transform(40:52,:),zeros(1,13),strcat('c','x'));
hold on

mat_test = [TEST{6}{1},TEST{6}{2},TEST{6}{3},TEST{6}{3}]
mat_test = mat_test'
mat_test_transform = mat_test*eig_v

subplot(1,2,2);
title('PCA for Testing Data');
hold on
plot(mat_test_transform(1:12,:),zeros(1,12),strcat('r','*'));
hold on
plot(mat_test_transform(13:24,:),zeros(1,12),strcat('g','o'));
hold on
plot(mat_test_transform(25:36,:),zeros(1,12),strcat('b','+'));
hold on
plot(mat_test_transform(37:48,:),zeros(1,12),strcat('c','x'));
hold on