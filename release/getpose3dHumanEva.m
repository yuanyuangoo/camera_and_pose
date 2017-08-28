setup
load mocapReducedModel.mat;
load pose.mat
im=zeros(640,480);
error=cell(size(poses,1),1);
%startmatlabpool(4);
pose3d=cell(size(poses,1),1);
string={'S1'    ,'Walking';
        'S1'	,'Jog';
        'S1'	,'ThrowCatch';
        'S1'	,'Gestures';
        'S1'	,'Box';
        'S2'	,'Walking';
        'S2'	,'Jog';
        'S2'	,'ThrowCatch';
        'S2'	,'Gestures';
        'S2'	,'Box';
        'S3'	,'Walking';
        'S3'	,'Jog';
        'S3'	,'ThrowCatch';
        'S3'	,'Gestures';
        'S3'	,'Box'};

for j=1:size(poses,1)
    posefile=[string{j,2},'_1_(C1)_',string{j,1},'.mat'];
    disp([num2str(j),'         ',posefile])
    xy=load(posefile);
    xy=xy.xy;
    xyz=zeros(15,3,size(xy,1));

%    pose=poses{j};
%    if isempty(pose)
%        error{j}=inf;
%    else
%        e=zeros(size(pose,3),1);
     for i=1:size(xy,1)
         disp([num2str(j),'         ',posefile,'          ',num2str(i)])
         [X, R, t] = recon3DPose(im,squeeze(xy(i,:,:)),'viz',0);
         xyz(:,:,i)=X;
         save;
     end
     pose3d{j}=xyz;

%           if all(pose(:,:,i) == 0)
%               e(i)=inf;
%            else
%                e(i)=geterror(pose(:,:,i),X,skel);
      
%            disp(e(i))
        
%        error{j}=e;
end
%closematlabpool

save('error.mat',error);
