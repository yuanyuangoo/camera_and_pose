setup
load mocapReducedModel.mat;
load data/H36m2d.mat
load data/H36m2ddelete.mat
load data/pose3d_h36m.mat
im=zeros(1002,1000);
sizeoferror=500;
r = randperm(size(ind,1),sizeoferror);
error=zeros(sizeoferror,1);
%startmatlabpool(4);
pose3d=zeros(sizeoferror,1);
xyz=zeros(15,3,sizeoferror);
parfor i=1:sizeoferror
    i
    xy1=xy(ind(r(i)),:,:);
    [X, R, t] = recon3DPose(im,squeeze(xy1),'viz',0);
    xyz(:,:,i)=X;
    %ground=squeeze(pose3d_valid(ind(r(i)),:,:));
    %error = geterror( ground,X,skel );
end