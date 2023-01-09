function v = get_nii_ROImean(input_img,ref_nii,fname,saveflag)

addpath('./Toolbox/NIfTI_20140122');
path='./resultdir/';

% input_img: .nii image
% ref_nii: brain parcellation .nii image
% fname: preferred name for the new image
% saveflag: save the new image (1) or not (0)

%%%%%%%%%%%%%%%%%%%%

switch nargin
    case 2
        fname='empty';
        saveflag=0;
    case 3
        saveflag=1;
end


% The parcellation template, node definition
auntouch=load_nii(ref_nii); 
d1=size(auntouch.img);
new=auntouch;

inputimg=load_nii(input_img);
d2=size(inputimg.img);

nnode=new.hdr.dime.glmax;
if d1 ~= d2
    disp('dimension not match !');
    return;
end

% start plotting
new.img=new.img.*0;

for i=1:nnode
    ind_node=find(auntouch.img==i);
    v(i)=double(mean(inputimg.img(ind_node)));
    new.img(ind_node)=v(i);    
end

if saveflag~=0
save_nii(new,[path,fname,'.nii']);
disp(['save new image'])
end


