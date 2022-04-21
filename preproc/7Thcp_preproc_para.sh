#!/bin/bash
# v3.2

# parameters: 
# $1: subid, eg. 100307; $2: Cfgname, eg. DPARSFA.mat; $3: ROIfile, eg *.mat or *.nii
# $4 data_dir, $5 result_dir



wk_dir="/tmp/preproc_hcp"
start_dir="FunImgARW"


# set up working dirs for dparsf preproc if not exist
[ ! -d "$wk_dir" ] && mkdir -p "$wk_dir"
[ ! -d "$wk_dir/$start_dir" ] && mkdir -p "$wk_dir/$start_dir"    



# get funtional image and put into start_dir FunImg  
cp -r $4/$1 $wk_dir/$start_dir/



### execute matlab function to perform preproc (dparsf)
cfgname=\'$2\'
roifile=\'$3\'
wkdir=\'$wk_dir\'
startdir=\'$start_dir\'

echo $wkdir

/usr/bin/matlab96 -singleCompThread -batch main_preproc_7T\($1,$cfgname,$roifile,$wkdir,$startdir\)

cp -r $wk_dir/Results $5/

