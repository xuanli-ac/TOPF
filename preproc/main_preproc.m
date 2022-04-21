function [Error,Cfg]=main_preproc(dataname,subid,Cfgname,ROIfile,wkdir,startdir)
%dataname: tfMRI_MOTOR_RL, rfMRI_REST1_RL
%Cfgname: DPARSFA_*.mat
%startdir: FunImgARW
%ROIfile: *ROICell.mat or **.nii

addpath(genpath('/data/project/movies_extrct_diff/xli_localsync/Toolbox/DPABI_V5.0_201001'))
addpath('/data/project/movies_extrct_diff/xli_localsync/Toolbox/spm12')

fname = ['id_',dataname,'_',num2str(subid),'.txt'];
dlmwrite(fname,subid,'precision',6)

tmp = load(Cfgname);
Cfg = tmp.Cfg;
disp('configuration loaded')

if contains(dataname,'REST')
    Cfg.TimePoints = 1200;
    disp('time set')
else
    if contains(dataname,'LANGUAGE')
        Cfg.TimePoints = 316;
        disp('time set')
    else
        if contains(dataname,'MOTOR')
            Cfg.TimePoints = 284;
            disp('time set')
        else
            if contains(dataname,'SOCIAL')
                Cfg.TimePoints = 274;
                disp('time set')
            else
                if contains(dataname,'WM')
                    Cfg.TimePoints = 405;
                    disp('time set')
                end
                
            end
        end
    end
end
                       
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




