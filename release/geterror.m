function [ error ] = geterror( ground,pose,skel )
clf
grid on

%Index = [8, 1, 10, 11, 11, 12, 13, 14, 14, 15, 2, 3, 3, 4, 5, 7, 7, 6, 8, 9];
%Index = [1, 2,  3,  4,  5,  6,  7,  8,  9, 10,11,12,13,14,15,16,17,18,19,20];
%Index=[2,11,12,14,15,18,16,19,20,3,4,6,7,8,10];
IndexofHumanEva=[2,11,12,14,15,16,18,19,20,3,4,6,7,8,10];
IndexofH36m=[1,2,3,4,5,6,7,9,11,12,13,14,15,16,17];
Index=IndexofHumanEva;
pv=centralize(pose);
gv=centralize(ground(Index,:));
z=[0,0,0];
scale=norm(gv);
S1=scale/norm(pv);
pv=pv*S1;
n=size(pv,1);
[ret_R,ret_t]=rigid_transform_3D(pv,gv);
A2 = (ret_R*pv') + repmat(ret_t, 1, n);
A2 = A2';

% Find the error
err = A2 - gv;
err = err .* err;
err = sum(err(:));
error = sqrt(err/n);

disp(sprintf("RMSE: %f", error));
disp("If RMSE is near zero, the function is correct!");

%visualizeGaussianModel(A2/100+[20,0,0],skel);
%visualizeGaussianModel(pvr/100+[10,0,0],skel);
%visualizeGaussianModel(gv/100,skel);


function S = centralize(S)

S = bsxfun(@minus,S,S(1,:));

function [mo,I] = getmo(S)
mo=zeros(size(S,1),1);
for i=1:size(S,1)
    mo(i)=sqrt(sum(S(i,:).^2));
end
[mo,I]=max(mo);
function Xr=rotate(X)
M1 = vrrotvec2mat(vrrotvec([0,1,0],X(8,:)));
i=9;
x=X*M1;
a=[x(i,1),0,x(i,3)];
b=[-1,0,0];
M2 = vrrotvec2mat(vrrotvec(b,a));
Xr=x*M2;

function dist = computeError3D(S1,S2)
dist=sqrt(sum((S1-S2).^2,2));
