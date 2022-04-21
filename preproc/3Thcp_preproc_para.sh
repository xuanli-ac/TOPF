#!/bin/bash
# v3.2

# parameters: $1: data, eg. tfMRI_MOTOR_RL; 
# $2: subid, eg. 100307; $3: Cfgname, eg. DPARSFA.mat; $4: ROIfile, eg *.mat or *.nii


data_dir="/tmp/hcp"
wk_dir="/tmp/preproc_hcp"
start_dir="FunImgARW"
result_dir="/data/project/TOPF/preproc/179Movie"
wmeanfile="/data/project/TOPF/preproc/wmean.nii.gz"

### datalad get data (give hcp s3 id and password)
export DATALAD_hcp_s3_key_id=id
export DATALAD_hcp_s3_secret_id=password

datalad_repo="https://github.com/datalad-datasets/human-connectome-project-openaccess.git"


# install dataset if it doesn't exist
[ ! -d "$data_dir" ] && datalad clone $datalad_repo $data_dir

# set up working dirs for dparsf preproc if not exist
[ ! -d "$wk_dir" ] && mkdir -p "$wk_dir"
[ ! -d "$wk_dir/$start_dir" ] && mkdir -p "$wk_dir/$start_dir"    
[ ! -d "$wk_dir/RealignParameter" ] && mkdir -p "$wk_dir/RealignParameter"
[ ! -d "$wk_dir/T1ImgNewSegment" ] && mkdir -p "$wk_dir/T1ImgNewSegment"



# get funtional image and put into start_dir FunImg  
datalad -C $data_dir get HCP1200/$2/MNINonLinear/Results/$1/$1.nii.gz
mkdir -p $wk_dir/$start_dir/$2
cp $data_dir/HCP1200/$2/MNINonLinear/Results/$1/$1.nii.gz $wk_dir/$start_dir/$2/

# get motion regressors, rename rp.txt, put into RealignParameter
datalad -C $data_dir get HCP1200/$2/MNINonLinear/Results/$1/Movement_Regressors.txt
mkdir -p $wk_dir/RealignParameter/$2
cp $data_dir/HCP1200/$2/MNINonLinear/Results/$1/Movement_Regressors.txt $wk_dir/RealignParameter/$2/rp.txt

# cp wmean file precomputed, will not be used but need to be there
cp $wmeanfile $wk_dir/RealignParameter/$2/


# get CSF and WM mask, put into T1ImgNewSegment 
datalad -C $data_dir get HCP1200/$2/MNINonLinear/ROIs/CSFReg.2.nii.gz
mkdir -p $wk_dir/T1ImgNewSegment/$2
cp $data_dir/HCP1200/$2/MNINonLinear/ROIs/CSFReg.2.nii.gz $wk_dir/T1ImgNewSegment/$2/wc3.nii.gz
datalad -C $data_dir get HCP1200/$2/MNINonLinear/ROIs/WMReg.2.nii.gz
cp $data_dir/HCP1200/$2/MNINonLinear/ROIs/WMReg.2.nii.gz $wk_dir/T1ImgNewSegment/$2/wc2.nii.gz


# set results folder in the project folder
[ ! -d "$result_dir/$1/Results" ] && mkdir -p "$result_dir/$1/Results"
[ ! -d "$result_dir/$1/Results/FD" ] && mkdir -p "$result_dir/$1/Results/FD"

# get head FD
datalad -C $data_dir get HCP1200/$2/MNINonLinear/Results/$1/Movement_RelativeRMS_mean.txt
cp $data_dir/HCP1200/$2/MNINonLinear/Results/$1/Movement_RelativeRMS_mean.txt $result_dir/$1/Results/FD/$2.txt

### execute matlab function to perform preproc (dparsf)
dataname=\'$1\'
cfgname=\'$3\'
roifile=\'$4\'
wkdir=\'$wk_dir\'
startdir=\'$start_dir\'
echo $dataname
echo $wkdir

/usr/bin/matlab96 -singleCompThread -batch main_preproc\($dataname,$2,$cfgname,$roifile,$wkdir,$startdir\)

cp -r $wk_dir/Results $result_dir/$1/
datalad -C $data_dir drop HCP1200/$sub/MNINonLinear
