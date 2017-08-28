error=cell(size(HumanEva3d,1),1);

for j=1:size(error,1)
    xyz=HumanEva3d{j};
    ground=HumanEva3dGround{j};
    e=zeros(size(xyz,3),1);
    parfor i=1:(size(xyz,3)-1)
        disp([i,j])
        if sum(sum(sum(ground(:,:,i))))~=0
            e(i)=geterror(ground(:,:,i),xyz(:,:,i),skel)
        else
            e(i)=inf
        end
    end
    error{j}=e;
end
%closematlabpool;