import sys
import os
import pandas as pd
import numpy as np
import scipy.stats
import scipy.io as sio
from julearn.utils import configure_logging
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

## configure_logging_julearn
configure_logging(level='INFO')


   
################################ Read arguments #################################################

# read input parameters (if Juseless,or terminal)
# print sys.argv[0] # prints python_script.py


wkdir = sys.argv[1]                     # path to project folder
sidfile = sys.argv[2]                   # path to subject id list
r_rootdir = sys.argv[3]                 # specify a result folder

nroi = int(sys.argv[4])                 # number of rois/parcels used, e.g., 268 for Shen atlas
seed = int(sys.argv[5])                 # a seed (int) for shuffling in cross validations
condind = int(sys.argv[6])              # which fMRI paradigm, index: 1-7
phenostr = sys.argv[7]                  # phenotype, a string
cutntr = int(sys.argv[8])               # preserve the first cutntr TRs of the given data, 0: use full length
feature_type = sys.argv[9]              # 'singlePC' or 'combinedPC'
pcind = int(sys.argv[10])               # >=1, pc index
threshold = float(sys.argv[11])         # feature selection by captured variance, e.g., 0.02

clfname = sys.argv[12]                  # ML model name, eg 'ridge', as defined by Julearn (https://juaml.github.io/julearn/main/steps.html)
setting_file = sys.argv[13]             # Path to the setting.txt file, which defines ML model parameters



if cutntr <=0:
    ntr = None  # full length 
else:
    ntr = cutntr # cut to cutntr

conditionlist = ['two_men','bridgeville','pockets','tfMRI_MOTOR_RL','tfMRI_SOCIAL_RL', 'tfMRI_WM_RL', 'tfMRI_LANGUAGE_RL']
#conditionlist = ['tfMRI_WM_LR']
fmricondition = conditionlist[condind-1]

# adding the core functions for TOPF (i.e., TOPF.py)
import TOPF



################################ Global variables ############################################

# load dataframe of each fMRI paradigms in datadir
datadir = wkdir+'/data/dataframe'
df_fmri = pd.read_csv(datadir+'/df_X_'+fmricondition+'.csv')

# 7T subjects family structure (restricted info): downloaded from http://db.humanconnectome.org/
df_family_info = pd.read_csv(datadir+'/RESTRICTED_7T.csv')

# 7T subjects phenotype info (unrestricted info): downloaded from http://db.humanconnectome.org/
df_pheno = pd.read_csv(datadir+'/Subjects_Info.csv')

# Familiy info files created by "create_files_familyinfo.m", subject index from zero 
famidfile=datadir+'/famid.txt'
Famid = np.loadtxt(famidfile)
df_F = pd.read_csv(datadir+'/FamilyIDinfo.csv')
Fmem = df_F[['FamilyMembers_1','FamilyMembers_2','FamilyMembers_3','FamilyMembers_4']].values


# check if subjects have missing data
subject_list = np.loadtxt(sidfile)
subject_list = subject_list.astype(int)
nsub = subject_list.shape[0]
print("number of subjects: ", nsub)
missub,newsubs = TOPF.check_missing_data_phenotype(df_pheno, subject_list, phenostr)
if missub:
    print('Missing values! Please adjust the subject_list.txt file')
    quit()

# other settings
k_outer =10
k_inner = 5

##################### set up result directory structure and file names
rfolder = f'/{phenostr}/cut_{str(ntr)}_t{threshold}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}'
rdir = r_rootdir+rfolder

if not os.path.exists(f"{rdir}"):
    os.makedirs(f"{rdir}")

Result_struct = TOPF.init_result_dir(rdir)

# for this older version, the features for different phenos are the same
#feature_dir = r_rootdir +f'/features/cut_{str(ntr)}_t{threshold}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}/'
feature_dir = r_rootdir +f'/features/cut_{str(ntr)}_ko{k_outer}_ki{k_inner}_n{nroi}/'

# save only for single PC
if not os.path.exists(f"{feature_dir}"):
    os.makedirs(f"{feature_dir}")

#################################################### START ###################################

print('current condition:', fmricondition)
print('number of rois used:', nroi)

# set up machine learning model
J_model = TOPF.create_ML_julearn_model(clfname, setting_file)

# start prediction: 
df_test, df_train, df_bp = TOPF.main_TOPF_kfold(df_fmri, fmricondition, clfname, nroi, seed, subject_list, df_pheno, Famid, Fmem, phenostr, J_model, Result_struct, feature_dir, feature_type, pcind, ntr, threshold, k_inner=5, k_outer=10, flip=None, clean=None, norm=1)

# calculate scores (defined in metric_list) for test data
nfold = k_outer
foldwise = 1  # compute the score for each fold separately

# save scores to scoredir
scoredir = Result_struct.scoredir
savepath = eval(Result_struct.scorefname_test)


seedlist = [seed] # needs to be a list
probtype = J_model.probtype
metric_list = J_model.metric_list
mean_all, mean_seed, df_measure, _ = TOPF.main_compute_prediction_scores(fmricondition, clfname, probtype, Result_struct, metric_list, seedlist, nfold, savepath, foldwise, test=True)

# print
print('current condition:', fmricondition)
print('scores of each fold:')
print(df_measure)

