setup

error=zeros(size(H36m3d,3),1);

for i=1:size(error,1)
    ind=indexall(indexchosen(i));
    xyz=H36m3d(:,:,i);
    ground=squeeze(H36m3d_ground_valid(ind,:,:))';
    disp(i)
    if sum(sum(sum(ground)))~=0
        error(i)=geterror(ground,xyz,skel);
    else
        error(i)=inf;
    end
end
%closematlabpool;