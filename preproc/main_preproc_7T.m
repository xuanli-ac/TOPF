function [Error,Cfg]=main_preproc_7T(subid,Cfgname,ROIfile,wkdir,startdir)

%Cfgname: DPARSFA_*.mat
%startdir: FunImgARW
%ROIfile: *ROICell.mat or **.nii

addpath(genpath('/data/project/movies_extrct_diff/xli_localsync/Toolbox/DPABI_V5.0_201001'))
addpath('/data/project/movies_extrct_diff/xli_localsync/Toolbox/spm12')

fname = ['id_',num2str(subid),'.txt'];
if ~exist(fname)
    dlmwrite(fname,subid,'precision',6)
end

tmp = load(Cfgname);
Cfg = tmp.Cfg;
disp('configuration loaded')

                       
Cfg.WorkingDir = wkdir;
Cfg.StartingDirName = startdir;

% parcellation
if contains(ROIfile,'.mat')
        tmp2=load(ROIfile);
        Cfg.CalFC.ROIDef=tmp2.ROICell;
        disp('parcellation set')
else
    if contains(ROIfile,'.nii')
        Cfg.CalFC.ROIDef={ROIfile};
        disp('parcellation set')
    end
end

        

%run preprocessing
[Error]=DPARSFA_run(Cfg,wkdir,fname,0);
disp('finish processing')




