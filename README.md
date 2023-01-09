# Topography-based predictive framework (TOPF)
>Code for implementing TOPF and relevant analysis

## Descriptions
### data folder
* SubjectID*.txt - IDs of the HCP subjects included in the dataset1 and dataset2 (described in the manuscript) separately
* example_dataframe/df_X_*.csv - an example dataframe of input fMRI data, which concatenates fMRI time series of all ROIs across subjects
* Subjects_info.csv - unrestricted phenotype information, downloaded from http://db.humanconnectome.org/
* create_files_familyinfo - code for producing the files regarding family structure information used in run_TOPF.py. This code requires restricted information of subjects, which can be accessed and downloaded from http://db.humanconnectome.org/ after application.

### preproc 
* custom bash and Matlab code for further preprocessing fMRI data

### code folder
* TOPF.py - core functions of TOPF, called by run_TOPF.py
* run_TOPF.py - python code to run TOPF
* run_TOPF.sh (wrapper function) + run_TOPF_submit.sh - if use clusters + HTCondor

## run TOPF:

Usage:
```
python run_TOPF.py $wkdir $subid_list $resultdir $nroi $seed $condition_index $phenotype_str $nTR $feature_type $pcindex $thre $clfname $settingfile_path

required arguments:
$wkdir: path to project folder
$subid_list: path to the file of subjects id list
$resultdir: path to result folder that will save outputs
$nroi: total number of rois used
$seed: an int used for shuffling samples during cross validations
$condition_index: 1-7, correspond to the seven fMRI conditions studied in the manuscript,  "conditionlist = ['two_men','bridgeville','pockets','tfMRI_MOTOR_RL','tfMRI_SOCIAL_RL', 'tfMRI_WM_RL', 'tfMRI_LANGUAGE_RL'], respectively
$phenotype_str: the label of the phenotype to be predicted given by HCP, e.g., PMAT24_A_CR
$nTR: length of data to be used. nTR= 170: use only the first 170 TRs of the given data; nTR=0: use all TRs without removing any (full length)
$feature_type: 'singlePC' or 'combinedPC'
$pcindex: int, pcindex = 1: use PC1 loadings as features; pcindex = 2 and feature_type = combinedPC: use PC1 and PC2 loadings as features
$thre: threshold of captured variance for feature selection
$clfname: machine learning (ML) algorithm name. All available clfnames used by Julearn can be seen here: https://juaml.github.io/julearn/main/steps.html
$settingfile_path: path to the file of ML algorithm settings, which specifies hyperparameters used by the chosen algorithm defined by scikit-learn
```
Examples:

```
in PC terminal:
python run_TOPF.py /myproject/TOPF /myproject/TOPF/data/SubjectID_Dataset2_HCP7T_179.txt /myproject/TOPF/results 268 0 1 PMAT24_A_CR 170 'singlePC' 1 0.02 ridge /myproject/TOPF/code/ridge_settings.txt

on cluster + HTCondor:
./run_TOPF_submit.sh | condor_submit
```




