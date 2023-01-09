import os
import ast
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_curve, roc_auc_score, precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold,KFold,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.io as sio
from julearn import run_cross_validation
from julearn.utils import configure_logging
import pickle
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from joblib import dump, load
from sklearn.metrics import explained_variance_score



###################################### Useful sub-functions ##########################################

##################################### some check-ups #############################
def check_missing_data_phenotype(df_pheno, subjects_id, phenostr):
    df_temp = df_pheno.loc[df_pheno['Subject'].isin(subjects_id)]
    S_temp = df_temp[['Subject',phenostr]]
    if S_temp[phenostr].isnull().values.any():
        print('please check you subjects, containing missing values for', phenostr)
        ind = S_temp[phenostr].isnull().values
        missing_subs = subjects_id[ind]
        new_subs = subjects_id[~ind]
        print("Values for these subjects are missing:", missing_subs)
    else:
        print('data are complete for Machine Learning part')
        missing_subs = []
        new_subs = []
    return missing_subs, new_subs
#####################################################################################

##################################### feature extraction and preprocessing ###############################

# could later move the discard trs here
def form_data_across_subjects(df_m, subjects_id, roi, ntr=None):
    """
    Aggregate z-score normalized fMRI time series across subjects within a given ROI.
    The returned Rsig is the input of PCA (feature extraction)
    #df_m: dataframe of fmri data from a given movie clip/run/session
    #ntr: desired number of trs to preserve
    #roi: an integer, roi index from 1!!
    #subjects_id: subjects to be analyzed
    """
    
    # get movie length from the first subject
    movie_len = df_m[df_m.subject==int(subjects_id[0])].shape[0]

    # init the data matrix for the roi
    X = np.empty((movie_len,0))

    # concatenate subjects within the given roi
    for sub in subjects_id:
        # check if data dimension is correct for each subject
        if df_m[df_m.subject == int(sub)].shape[0]:
            df_temp = df_m[df_m.subject==int(sub)]
            x_temp = df_temp[str(roi)].values
            x_temp = np.reshape(x_temp,(movie_len,1))
            X = np.append(X, x_temp, axis = 1)
        else:
            print('fmri data of subject ', sub, ' mismatch with others!')
            break

    # truncate the data to the desired length if needed
    # from the first to the ntr-th volume  
    X = X[0:ntr,:]
        
    # z-score normalise each time series, ddof=1 consistent with matlab
    X = scipy.stats.zscore(X,ddof=1)  # nTR * nSub
    return X


def normalise_feature(F,Ft):
    """
    normalise feature for ML for training and test separately to avoid data leakage
    F: training; ntrainsub*nfeatures
    Ft: test; ntestsub*nfeatures
    """
    scaler = StandardScaler().fit(F)
    F = scaler.transform(F)
    Ft = scaler.transform(Ft)
    return F, Ft   


def flip_pc_and_clean_loadings(pc_score,pc_loading,flip=None,clean=None):
    """
    flip pc score and loadings when needed, eg. for cross-condition comparison
    """
    if flip =='abs':
        # make sure the maximum absolute value comes from postive
        if np.abs(pc_loading).max()>np.max(pc_loading):
            pc_loading = pc_loading*-1
            pc_score = pc_score*-1
    elif flip == 'num':
        # make sure number of pos larger than number of neg
        if np.sum(pc_loading >= 0) < np.sum(pc_loading < 0):
            pc_loading = pc_loading*-1
            pc_score = pc_score*-1
    else:
        # do nothing
        pc_score = pc_score
        pc_loading = pc_loading
    
    # set all negative to zero
    if clean:
        pc_loading[np.argwhere(pc_loading<0)] = 0 
    
    return pc_score,pc_loading

def feature_selection(train_data, test_data, Var, pcind, phenostr, threshold):
    # select features according to threshold
    Var = np.reshape(Var,[-1,1])
    roi_preserved = np.argwhere(Var>=threshold)[:,0]
    roi_labels = list(roi_preserved+1)  # set variable names for features
    X_keys = [f'%d_PC{pcind}' % i for i in roi_labels]
    columns = ['Subject']+[phenostr]+X_keys
    train_data = train_data[columns]
    test_data = test_data[columns]
    return train_data, test_data, X_keys


def perform_feature_extraction_singlePC(df_m, sub_train, sub_test, nroi, pcind=1, ntr=None, flip=None, clean=None, norm=1):
    """
    # Extract and normalise features for training and testing subjects of each outer CV fold.
    ### Features refer to the individual-specific topographies.
    ### For training subjects, each feature is the PC loading of a ROI.
    ### For testing subjects, each feature is the correlation between PC learned on training and fMRI time series of a ROI.
    ### sub_train/sub_test: subjects' ids for training/test sets; e.g. 100610
    ### nnode: number of rois
    ### ntr: desired number of TRs to preserve; default preserve all TRs
    ### pcind: index of PC whose loadings will be used as features, index from 1 
    ### F, Ft, Var: Features for training, test and variance explained by PC
    ### flip = 'abs','num' if flip pc loadings and scores
    ### clean = 1, if set all neg loadings to pos
    ### norm = 1, z-score normalise each feature
    """
    
    nc = pcind+1 # nc: keep the first nc components (PC)
    ntrain = sub_train.shape[0]
    ntest = sub_test.shape[0]
    
    # get movie_len and check ntr
    movie_len = df_m[df_m.subject==int(sub_train[0])].shape[0] # full length

    if ntr is not None:
        if ntr>movie_len:
            print('desired TRs larger than possible, using the full length!')
            ntr = movie_len
    else:
        ntr = movie_len #tranform none to a integer
    
    print('Number of TRs used: ', ntr)    
    
    F = np.empty([ntrain,0],float) # training features
    Ft = np.empty([1,0],float) # testing features
    Var = np.empty([1,0],float) # variance explained by PC
    PCs = np.empty([ntr,0],float) # PC scores; shared responses
    
    pca = PCA(n_components=nc)
    
    # TRAINING: perform PCA for each ROI and extract PC loadings as features
    # roi index from 1 !!
    for roi in range(1,nroi+1,1):
        X = form_data_across_subjects(df_m, sub_train, roi, ntr)
        SCORE = pca.fit_transform(X) #column
        COEFF = pca.components_
        EXPLAINED = pca.explained_variance_ratio_
        LATENT = pca.explained_variance_
        
        # get pc loadings and scores
        temp = COEFF[pcind-1,:]*np.sqrt(LATENT[pcind-1]) # PC_pcind loading as row: pcind=1 refers to PC1
        temp = np.reshape(temp, (-1, 1)) # change to column
        stemp = SCORE[:,pcind-1]
        stemp = np.reshape(stemp,(-1,1))
    
        # flip and clean if needed
        stemp,temp = flip_pc_and_clean_loadings(stemp,temp,flip,clean)

        # save 
        F = np.append(F,temp,axis=1)   # pc_score: ntrain*nroi
        Var = np.append(Var,EXPLAINED[pcind-1]) # variance
        PCs = np.append(PCs, stemp, axis=1) #loading: movie_len or ntr*nroi
        
    # Testing:
        Xt = form_data_across_subjects(df_m, sub_test, roi, ntr)
        for testind in range(0,ntest,1):
            temp = scipy.stats.pearsonr(Xt[:,testind],SCORE[:,pcind-1]) # PC_pcind score
            tr = temp[0] # correlation = loading
            # set to zero if negative if clean != 0
            if clean:
                if tr<0:
                    tr = tr*-1
            Ft = np.append(Ft,tr)

    # print process:        
        if roi % 50 == 0:
            print('Node',roi,' PCA finished')

    Ft = np.reshape(Ft, (-1, ntest)) 
    Ft = np.transpose(Ft) # (ntest,nroi)
    Var = np.reshape(Var,[-1,1])
   
    # set variable names for all features
    X_keys = [f'%d_PC{pcind}' % i for i in range(1, nroi+1, 1)]

    # normalizing each feature for training and test
    if norm:
        F,Ft = normalise_feature(F,Ft)
    
    return F,Ft,Var,PCs,X_keys


def perform_feature_extraction_multiplePC(df_m, sub_train, sub_test, nroi, pcind=1, ntr=None, flip=None, clean=None, norm=1):
    """
    Extract features for training and testing subjects of each outer CV fold.
    Features refer to the individual-specific topographies.
    For training subjects, each feature is the PC loading of a ROI.
    For testing subjects, each feature is the correlation between PC learned on training and fMRI time series of a ROI.
    # sub_train/sub_test: subjects' ids for training/test sets; e.g. 100610
    # nnode: number of rois
    # ntr: desired number of TRs to preserve; default preserve all TRs
    # npc: number of PCs whose loadings will be used as features
    # F, Ft, Var: Features for training, test and variance explained by PC
    """

    ntrain = sub_train.shape[0]
    ntest = sub_test.shape[0]

    F = np.empty([ntrain,0],float) # training features
    Ft = np.empty([ntest,0],float) # testing features
    Var = np.empty([nroi,0],float) # variance explained by PC
    PCs = np.empty([0,nroi],float) # PC scores; shared responses
    X_keys = []

    for pc in range(1,pcind+1,1):
        F_temp,Ft_temp,Var_temp,PCs_temp,Xk_temp = perform_feature_extraction_singlePC(df_m, sub_train, sub_test, nroi, pc, ntr, flip, clean, norm)
        F = np.append(F,F_temp,axis=1)
        Ft = np.append(Ft,Ft_temp,axis=1)
        Var = np.append(Var,Var_temp,axis=1)
        if pc == 1:
            PCs = np.append(PCs,PCs_temp,axis=0)
        else:
            PCs = np.append(PCs,PCs_temp,axis=1)
        X_keys = X_keys+Xk_temp

    return F,Ft,Var,PCs,X_keys



def get_dataframe(F,sub_list,df_pheno,phenostr,X_keys): 
    """
    ## get the targets of train/test subjects and create dataframe (subid+target+features) for ML (julearn)
    Integrate features and target (phenotype) as a single dataframe ready for prediction
    Subjects with missing values should be excluded beforehand by function "check_missing_data_phenotype"
    """  
    # df_pheno: phenotype information provided by HCP ('unrestricted')
    # sub_train/sub_test: sub id
    # phenostr: the HCP label for the phenotype to be predicted
    
    # get target scores for the given subjects and phenotype
    df_temp = df_pheno.loc[df_pheno['Subject'].isin(sub_list)] # data for training subjects
    df_score = df_temp[['Subject', phenostr]].sort_values(by='Subject',kind='stable').reset_index(drop=True)   
    
    # convert features to dataframe
    df_F = pd.DataFrame(data=F,columns=X_keys)
  
    # Insert 'Subject' column
    df_F.insert(0, 'Subject', sub_list)
   
    # combine feature and target according to subject id in target (sorted already)
    df_data = pd.merge(df_score, df_F, on="Subject", how="left")

    return df_data


def perform_permutation_train(train_data,X_keys,y_keys):
    
    y_train = train_data[y_keys].values
    X_train = train_data[X_keys].values
    y_train = np.reshape(y_train, (-1, 1))
    
    ns = y_train.shape[0] # number of training subjects
    indp = np.random.permutation(ns)
    y_train = y_train[indp,:]
    
    d1 = np.concatenate([X_train, y_train], axis=1)
    vna = X_keys+[y_keys]
    train_data = pd.DataFrame(data=d1, columns=vna)
    
    return train_data

def check_if_features_complete(pcind, feature_dir, Result_struct, fmricondition, seed, foldind):
    """
    check if all singlePC files exist for computing combinedPC
    """
    missing = []
    for i in range(1,pcind+1):
        featuredir = feature_dir+f'PC{i}'+'/'
        # feature file names
        ftestfile = eval(Result_struct.featurefname_test)
        ftrainfile = eval(Result_struct.featurefname_train)
        varfname = eval(Result_struct.varfname)

        # if exist, read in
        if os.path.isfile(ftestfile) and os.path.isfile(ftrainfile) and os.path.isfile(varfname):
            print('Features for pc',i,'complete')
        else:
            missing = missing + [i]
    return missing
                    
####################################################################################################################



############################################ machine learning set up ###############################################

##################################### This block will not be used in the future versions ##########################
def cv_outer_kfold(seedind, k, Famid):
    """
    This function will not be used anymore, instead, cv_control_for_family will be used.
    """
    # Outer k-fold cross-validation, splitting according to Family structure
    if seedind == 0:
        kf = KFold(n_splits=k, shuffle=False)
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=seedind)
    cv_fit = kf.split(Famid)    

    return cv_fit

def get_sub_index_eachfold(subid, train_index, test_index, Fmem):
    """
    This function will not be used anymore, instead, cv_control_for_family will be used.
    """
    #select the subjects in the selected families for both train and test
    temp = np.reshape(Fmem[train_index],(-1,1))
    ind = np.argwhere(~np.isnan(temp)) # get sub id num
    ind = ind[:,0] #
    temp = temp[ind,:]
    train_index=temp[:,0] # (ntrain,) subjects index
    train_index = train_index.astype(int)
        
    temp = np.reshape(Fmem[test_index],(-1,1))
    ind = np.argwhere(~np.isnan(temp)) # get sub id num
    ind = ind[:,0] #
    temp = temp[ind,:]
    test_index=temp[:,0] # (ntest,)
    test_index = test_index.astype(int)
        
    ######### change index to subject ids 11.11.2022
    sub_train = subid[train_index]
    sub_test = subid[test_index]
    #########

    return sub_train, sub_test

def main_combine_feature_dataframes(df_m, fmricondition, seed, foldind, feature_dir, Result_struct, df_pheno, phenostr, sub_train, sub_test, nroi, pcind, ntr, missing=None, threshold=0, flip=0, clean=0, norm=1):
    
    ntrain = sub_train.shape[0]
    ntest = sub_test.shape[0]

    F = np.empty([ntrain,0],float) # training features
    Ft = np.empty([ntest,0],float) # testing features
    X_keys = []
    
    for i in range(1,pcind+1):
        featuredir = feature_dir +f'PC{i}'+'/'
        ftestfile = eval(Result_struct.featurefname_test)
        ftrainfile = eval(Result_struct.featurefname_train)
        varfname = eval(Result_struct.varfname)
        if i in missing:
            F_temp,Ft_temp,Var,PCs,Xk_temp = perform_feature_extraction_singlePC(df_m, sub_train, sub_test, nroi, pcind, ntr, flip, clean, norm)
            epvar = pd.DataFrame(data=Var) # all rois for each PC
            epvar.to_csv(varfname)
            train_data = get_dataframe(F_temp,sub_train,df_pheno,phenostr,Xk_temp)
            test_data = get_dataframe(Ft_temp,sub_test,df_pheno,phenostr,Xk_temp)
            test_data.to_csv(ftestfile, header=True,index=True)
            train_data.to_csv(ftrainfile, header=True,index=True)
        else:
            test_data = pd.read_csv(ftestfile)
            train_data = pd.read_csv(ftrainfile)
            Xk_temp = [col for col in test_data.columns if 'PC' in col]
            F_temp = train_data[Xk_temp].values
            Ft_temp = test_data[Xk_temp].values
            var_tmp = pd.read_csv(varfname)
            Var = var_tmp.iloc[:,1].values
        if threshold:
            Var = np.reshape(Var,[-1,1])
            roi_preserved = np.argwhere(Var>=threshold)[:,0]
            roi_labels = list(roi_preserved+1)  # set variable names for features
            Xk_temp = [f'%d_PC{i}' % l for l in roi_labels]
            print(F_temp.shape)
            F_temp = F_temp[:,roi_preserved]
            Ft_temp = Ft_temp[:,roi_preserved]
            print('PC',i,': Number of features used:', len(Xk_temp))
        X_keys = X_keys + Xk_temp
        F = np.append(F,F_temp,axis=1)
        Ft = np.append(Ft,Ft_temp,axis=1)

    train_data = get_dataframe(F,sub_train,df_pheno,phenostr,X_keys)
    test_data = get_dataframe(Ft,sub_test,df_pheno,phenostr,X_keys)

    return train_data, test_data, X_keys

#########################################################################################################



def cv_control_for_family(subject_list,df_pheno, phenostr, seed, k=10, df_family_info=None):
    """
    split train/test for cv folds and control for family structure if needed (for HCP!)
    subject_list: full subject list
    df_pheno: phenotype.csv
    phenostr: phenotype label of the given dataset
    df_family_info: gives the HCP family info
    """
    # target of all subjects
    df_target = df_pheno[['Subject', phenostr]]
    df_target = df_target[df_target.Subject.isin(subject_list)].sort_values(by='Subject',kind='stable').reset_index(drop=True)
    target = df_target[phenostr].values

    # family info; control for family structure
    if df_family_info is not None:
        df_fam = df_family_info[['Subject', 'Family_ID']]
        df_fam = df_fam[df_fam.Subject.isin(subject_list)].sort_values(by='Subject',kind='stable').reset_index(drop=True)
        groups = df_fam['Family_ID'].values
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        cv_fit = cv.split(subject_list, target, groups)

    # if no family info available   
    else:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        cv_fit = cv.split(subject_list, target)
        groups = None
    
    return cv_fit, groups

####### set up set up model paramters from setting files
class create_ML_julearn_model:
    def __init__(self, clfname, setting_file) -> None:
        
        # currently available models of Julearn: https://juaml.github.io/julearn/main/steps.html
        probtype_available = ['regression','binary_classification','multiclass_classification']
        clfnamelist = ['svm', 'rf', 'et', 'adaboost','baggging','gradientboost','gauss','logit','logitcv','linreg','ridge','ridgecv','sgd']

        # read in model settings from a setting.txt
        if os.path.exists(setting_file):
            with open(setting_file) as f:
                ms = f.read()
            model_params = ast.literal_eval(ms) # a dict

            # probtype
            key = 'problem_type'
            if key in model_params.keys():
                probtype = model_params[key]
                model_params.pop(key, None) # delete the key
                if any(probtype in s for s in probtype_available):
                    self.probtype = probtype
                else:
                    print('Specified problem type not found! Set to binary_classification automatically')
                    self.probtype = 'binary_classification'
            else:
                print('problem_type not defined! Set to binary_classification automatically')
                self.probtype = 'binary_classification'

            # metric_list
            key = 'metric_list'
            if key in model_params.keys():
                metric_list = model_params[key]
                model_params.pop(key, None) # delete the key
                self.metric_list = metric_list
            else:
                self.metric_list = []
                print('metric_list not defined! will use the default metrics according to problem_type later')
            
            # clfname
            if any(clfname in s for s in clfnamelist):    
                self.name = clfname
            else:
                print('Specified model not found! Set to svm automatically')
                self.name = 'svm'
                
            # the rest keys will be used in model_params
            if len(model_params.keys()) == len(model_params.values()):
                self.model_params = model_params
            else:
                print('number of model hyperparameters and number of given values mismatch!')

        else:
            print('please specify the path to model settings')
            

######################################################################################################################


########################################  set up results formation ###################################################

####### results folder structure set up
class init_result_dir:
    def __init__(self, rdir) -> None:
        # rdir: the resultdir specified beforehand, e.g.,

        ######################################## prediction dir 
        preddir = rdir+'/prediction/'
        if not os.path.exists(f"{preddir}"):
            os.makedirs(f"{preddir}")
        self.preddir = preddir

        # file names for predictions on test data/train data over all folds
        self.predfname_test = "preddir + f'test_{clfname}_{fmricondition}_seed{seed}.csv'"
        self.predfname_train = "preddir + f'train_{clfname}_{fmricondition}_seed{seed}.csv'"
        
        
        ######################################## model dir 
        modeldir = rdir+'/fitted_models/'
        if not os.path.exists(f"{modeldir}"):
            os.makedirs(f"{modeldir}")
        self.modeldir = modeldir

        # save model of each fold separately
        self.modelfname = "modeldir + f'model_{clfname}_{fmricondition}_seed{seed}_fold{foldind+1}.sav'"
        self.modelfname_bp = "modeldir + f'bestparams_{clfname}_{fmricondition}_seed{seed}.csv'"
        
        ######################################## feature dir 
        featuredir = rdir+'/features/'
        if not os.path.exists(f"{featuredir}"):
            os.makedirs(f"{featuredir}")
        self.featuredir = featuredir
        
        #save feature dataframe of each fold separately
        self.featurefname_test = "featuredir + f'features_test_{fmricondition}_seed{seed}_fold{foldind+1}.csv'"
        self.featurefname_train = "featuredir + f'features_train_{fmricondition}_seed{seed}_fold{foldind+1}.csv'"
        self.varfname = "featuredir + f'var_{fmricondition}_seed{seed}_fold{foldind+1}.csv'"

        ######################################## score dir 
        scoredir = rdir+'/scores/'
        if not os.path.exists(f"{scoredir}"):
            os.makedirs(f"{scoredir}")
        self.scoredir = scoredir
        self.scorefname_test = "scoredir + f'/scores_test_{clfname}_{fmricondition}_seed{seed}.csv'"
        self.scorefname_train = "scoredir + f'/scores_train_{clfname}_{fmricondition}_seed{seed}.csv'" 


####### concatenate_best_params 
def concatenate_best_params(best_param, bp, foldind):
    # best_param: current dict to be updated
    # bp: from the given (foldind) fold 
    # foldind: 0 
    # merge two dictionaries
    def mergeDictionary(dict_1, dict_2):
        dict_3 = {**dict_1, **dict_2}
        for key, value in dict_3.items():
            if key in dict_1 and key in dict_2:
                dict_3[key] = [value , dict_1[key]]
        return dict_3
    
    if foldind == 0:
        best_param = bp
    elif foldind ==1:
        best_param = mergeDictionary(best_param, bp) #values -->list
    else:
        for key, value in bp.items():
            best_param[key]+=[value]
    return best_param
######################################################################################################################



############################################# calculate prediction scores #############################################
def get_predproba_for_roc_auc(estimator, J_model, test_data, X_keys):
    """
    get predict_proba on test for calculating roc_auc for binary classification if available
    """ 
    # estimator/model: fitted model (Julearn output)
    # J_model: class defined by create_ML_julearn_model, containing model parameters
    # test_data: feature dataframe
    # X_keys: feature labels
    # ytest_proba: output of predict_proba or decision_function of given samples

    if J_model.probtype == 'binary_classification':
        # svm or similar clfs, which set clf__probability = True 
        key_proba = [key for key, value in J_model.model_params.items() if 'probability' in key]
        if key_proba and J_model.model_params[key_proba[0]]:
            ytest_proba = estimator.predict_proba(test_data[X_keys])[:,1] #binary case: get probability of the class with "greater label"
        # random forest/or other similar clfs
        elif J_model.name!='svm' and hasattr(estimator, 'predict_proba') and callable(estimator.predict_proba):
            ytest_proba = estimator.predict_proba(test_data[X_keys])[:,1]
        # ridge and similar clfs
        elif hasattr(estimator, 'decision_function') and callable(estimator.decision_function):
            ytest_proba = estimator.decision_function(test_data[X_keys])
        else:
            ytest_proba = []
    else:
        ytest_proba = []
    return ytest_proba



def cal_measures(df_temp, probtype, metric_list):
    """
    calculate measures of a given prediction with true values using metrics in metric_list or use default if empty
    """
    # final metric_list and measures: just in case some predefined metrics can't be calculated
    mlist = []
    measures = []

    if df_temp.empty:
        print('Input dataframe is empty!')
            
    if probtype == 'binary_classification':
        # if metric_list is empty, set to default 
        if not metric_list:
            metric_list = ['accuracy','balanced_accuracy','roc_auc']
        for met in metric_list:
            if met == 'accuracy':
                acc = accuracy_score(df_temp.true, df_temp.pred)
            elif met =='balanced_accuracy':
                bacc = balanced_accuracy_score(df_temp.true, df_temp.pred)
            elif met == 'roc_auc':
                auc = roc_auc_score(df_temp.true, df_temp.proba)
            else:
                print('this measure needs to be manually defined in func "cal_measures" !')
        if acc:
            measures = measures + [acc]
            mlist = mlist + ['accuracy']
        if bacc:
            measures = measures + [bacc]
            mlist = mlist + ['balanced_accuracy']
        if auc:
            measures = measures + [auc]
            mlist = mlist + ['roc_auc']

    #### for regression not tested yet ###########
    elif probtype == 'regression':
        # if metric_list is empty, set to default 
        if not metric_list:
            metric_list = ['pearson','spearman','r2','mean_absolute_error']
        for met in metric_list:
            if "pearson" in met:
                pr = scipy.stats.pearsonr(df_temp.true, df_temp.pred)
            if "spearman" in met:
                sr = scipy.stats.spearmanr(df_temp.true, df_temp.pred)
            if "r2" in met:
                r2 = r2_score(df_temp.true, df_temp.pred)
            if met == "mean_absolute_error":
                mae = mean_absolute_error(df_temp.true, df_temp.pred)
                
        if pr:
            measures = measures + [pr[0]]
            mlist = mlist + ['pearson']
        if sr:
            measures = measures + [sr[0]]
            mlist = mlist + ['spearman']
        if r2:
            measures = measures + [r2]
            mlist = mlist + ['r2']
        if mae:
            measures = measures + [mae]
            mlist = mlist + ['mean_absolute_error']    
    
    return measures, mlist

######################################################################################################################



######################################################## MAIN FUNCTIONS OLDER - will not be used in the future ##############################################

def main_TOPF_kfold(df_m, fmricondition, clfname, nroi, seed, subject_list, df_pheno, Famid, Fmem, phenostr, J_model, Result_struct, feature_dir, feature_type='singlePC', pcind=1, ntr=None, threshold=0, k_inner=5, k_outer=10, flip=None, clean=None, norm=1):
    
    """
    J_model: self-defined ML Julearn model class. Parameters set outside beforehand. Could be replaced by sklearn models when needed
    feature_type: "singlePC" or "combinedPC". 
    pcind: when "singlePC", use the pcind-th PC loadings as features; when combinedPC, use the 1st to pcind-th PC loadings together as features
    ntr: use the first ntr TRs data (default = None, using the full length without truncation)
    """
    ######################################## create empty list to save results
    
    # for saving predictions
    pred_train = []  
    true_train = []
    pred_test = []
    true_test = []
    proba_test = [] # predict_proba on test
    
    fold_label_train = []
    fold_label_test = []
    seed_label_train = []
    seed_label_test = []
    subid_train = []
    subid_test = []
    best_param = {}

    # initialize result directory structures + filename patterns
    # Result_struct = init_result_dir(rdir) # done outside in the run.py
    
    modeldir = Result_struct.modeldir
    preddir = Result_struct.preddir
    
    
    
    ######################################## start outer cv 

    # get outer_cv splits (stratifiedgroupkfold) while controlling for family structure (HCP!)
    outer_cv = cv_outer_kfold(seed, k_outer, Famid)
    y_keys = phenostr

    # start outer_cv
    for foldind, (train_index, test_index) in enumerate(outer_cv):
        # print
        print('current fold: ', foldind+1)

        # train and test subject_id lists
        sub_train, sub_test = get_sub_index_eachfold(subject_list, train_index, test_index, Fmem)
        sub_train = np.sort(sub_train)
        sub_test = np.sort(sub_test)

        ######################################## get features and save dataframes
        if feature_type == 'singlePC':
            # feature file names
            featuredir = feature_dir + f'PC{pcind}'+'/' # shared feature_dir
            ftestfile = eval(Result_struct.featurefname_test)
            ftrainfile = eval(Result_struct.featurefname_train)
            varfname = eval(Result_struct.varfname)

            # if exist, read in
            if os.path.isfile(ftestfile) and os.path.isfile(ftrainfile) and os.path.isfile(varfname):
                print('Features already extracted and saved.')
                print('Test data path: ', ftestfile)
                print('Train data path: ', ftrainfile)
                test_data = pd.read_csv(ftestfile)
                train_data = pd.read_csv(ftrainfile)
                X_keys = [col for col in test_data.columns if 'PC' in col]
                Var = pd.read_csv(varfname)
                Var = Var.iloc[:,1].values
                # check if the targets match
                if y_keys not in train_data.columns:
                    F = train_data[X_keys].values
                    Ft = test_data[X_keys].values
                    train_data = get_dataframe(F,sub_train,df_pheno,phenostr,X_keys)
                    test_data = get_dataframe(Ft,sub_test,df_pheno,phenostr,X_keys)

            else:
                # otherwise, get features - PC loadings and save for later use
                F,Ft,Var,PCs,X_keys = perform_feature_extraction_singlePC(df_m, sub_train, sub_test, nroi, pcind, ntr, flip, clean, norm)
                train_data = get_dataframe(F,sub_train,df_pheno,phenostr,X_keys)
                test_data = get_dataframe(Ft,sub_test,df_pheno,phenostr,X_keys)
                test_data.to_csv(ftestfile, header=True,index=True)
                train_data.to_csv(ftrainfile, header=True,index=True)
                epvar = pd.DataFrame(data=Var) # all rois for each PC
                epvar.to_csv(varfname)
            
            # feature selection
            if threshold:
                train_data, test_data, X_keys = feature_selection(train_data, test_data, Var, pcind, phenostr, threshold)
            print('Number of features used:', len(X_keys))

        elif feature_type == 'combinedPC':
            # check if files of features for single PC exist - to speed up
            missing = check_if_features_complete(pcind, feature_dir, Result_struct, fmricondition, seed, foldind)
            
            # combine all features created before
            train_data, test_data, X_keys = main_combine_feature_dataframes(df_m, fmricondition, seed, foldind, feature_dir, Result_struct, df_pheno, phenostr, sub_train, sub_test, nroi, pcind, ntr, missing, threshold, flip, clean, norm)

        else:
            print('please specify feature type: "singlePC" or "combinedPC"!')   

        ######################################## settings for inner cv 
        
        
        
        ######################################## train model on training data using inner-cv settings

        # Note: this part could be replaced by sklearn functions if needed
        ##################################### Julearn specific ########################################################
        scores, estimator = run_cross_validation(
        X=X_keys, y=y_keys, data=train_data, preprocess_X=None,
        problem_type=J_model.probtype, model= J_model.name, model_params=J_model.model_params, return_estimator='final',
        cv=k_inner)
         ###############################################################################################################

    
        ######################################## save the best parameters and models of each fold
        
        # when tuning hyperparameters: should has attribute "best_estimator_"
        if hasattr(estimator, 'best_estimator_'):
            model = estimator.best_estimator_
            bp = estimator.best_params_
            print('best parameter: ', bp)
            best_param = concatenate_best_params(best_param, bp, foldind)
        else:
            print('Has no attribute best_estimator_')
            model = estimator
    
        ######################################## save fitted model
        # save to fitted_models folder
        filename = eval(Result_struct.modelfname)
        pickle.dump(model, open(filename, 'wb'))

        ######################################## save predictions of each fold

        # predict on test data
        ytest_pred = model.predict(test_data[X_keys])
        ytest = test_data[y_keys]

        # final training predictions
        ytrain_pred = model.predict(train_data[X_keys])
        ytrain = train_data[y_keys]

        # get predict_proba on test for calculating roc_auc for binary classification if available
        ytest_proba = get_predproba_for_roc_auc(model, J_model, test_data, X_keys)

        # Print out some scores just out of curiosity
        if J_model.probtype == 'binary_classification':
            bacc_test = balanced_accuracy_score(ytest, ytest_pred)
            bacc_train = balanced_accuracy_score(ytrain, ytrain_pred)
            print('Test balanced accuracy for fold', foldind+1, 'is', bacc_test)
            print('Training balanced accuracy for fold', foldind+1, 'is', bacc_train)
        elif J_model.probtype == 'regression':
            pr_test = scipy.stats.pearsonr(ytest, ytest_pred)
            pr_train = scipy.stats.pearsonr(ytrain, ytrain_pred)
            print('Test pearson correlation for fold', foldind+1, 'is', pr_test[0])
            print('Training pearson correlation for fold', foldind+1, 'is', pr_train[0])

        ######################################## concatenate results across all folds
        pred_train = pred_train + list(ytrain_pred)
        true_train = true_train + list(ytrain)

        pred_test = pred_test + list(ytest_pred)
        true_test = true_test + list(ytest)
        proba_test = proba_test + list(ytest_proba)

        #fold_label_train = fold_label_train + [f'fold{foldind+1}']*len(sub_train)
        fold_label_train = fold_label_train + [foldind+1]*len(sub_train)
        fold_label_test = fold_label_test + [foldind+1]*len(sub_test)
        
        seed_label_train = seed_label_train + [seed]*len(sub_train)
        seed_label_test = seed_label_test + [seed]*len(sub_test)

        subid_train = subid_train + list(sub_train)
        subid_test = subid_test + list(sub_test)
        

    ######################################## save prediction results as dataframes for training and test separately of all folds   
    Ptest = []
    Ptest.append(pred_test)
    Ptest.append(true_test)
    df_pred_test = pd.DataFrame(Ptest).T
    df_pred_test.columns = ['pred','true']
    if proba_test: 
        df_pred_test['proba'] = proba_test
    df_pred_test.insert(0,'Subject',subid_test)
    df_pred_test.insert(0,'fold',fold_label_test)
    df_pred_test.insert(0,'seed',seed_label_test)

    Ptrain = []
    Ptrain.append(pred_train)
    Ptrain.append(true_train)
    df_pred_train = pd.DataFrame(Ptrain).T
    df_pred_train.columns = ['pred','true']
    df_pred_train.insert(0,'Subject',subid_train)
    df_pred_train.insert(0,'fold',fold_label_train)
    df_pred_train.insert(0,'seed',seed_label_train)
    
    # save to prediction folder
    fileptest = eval(Result_struct.predfname_test)
    df_pred_test.to_csv(fileptest, header=True)
    fileptrain = eval(Result_struct.predfname_train)
    df_pred_train.to_csv(fileptrain, header=True)

    ######################################## save best parameters of all folds
    BP = []
    parlist = list(best_param.keys())
    for p in parlist:
        BP.append(list(best_param[p]))
    df_best_params = pd.DataFrame(BP).T
    df_best_params.columns = parlist
    print('best paramters:')
    print(df_best_params)
    df_best_params.insert(0,'fold', np.arange(1,k_outer+1,1))
    df_best_params.insert(0,'seed',[seed] * k_outer)
    #print(df_best_params)

    # save to model folder
    filebp = eval(Result_struct.modelfname_bp)
    df_best_params.to_csv(filebp, header=True, index=True)

    #return df_pred_test, df_pred_train, df_best_params
    return df_pred_test, df_pred_train, df_best_params

######################################################################################################################


######################################################## MAIN FUNCTIONS ##############################################

def main_TOPF(df_m, fmricondition, clfname, nroi, seed, subject_list, df_pheno, df_family_info, phenostr, J_model, Result_struct, feature_type='singlePC', pcind=1, ntr=None, threshold=0, k_inner=5, k_outer=10, flip=None, clean=None, norm=1):
    
    """
    J_model: self-defined ML Julearn model class. Parameters set outside beforehand. Could be replaced by sklearn models when needed
    feature_type: "singlePC" or "combinedPC". 
    pcind: when "singlePC", use the pcind-th PC loadings as features; when combinedPC, use the 1st to pcind-th PC loadings together as features
    ntr: use the first ntr TRs data (default = None, using the full length without truncation)
    """
    ######################################## create empty list to save results
    
    # for saving predictions
    pred_train = []  
    true_train = []
    pred_test = []
    true_test = []
    proba_test = [] # predict_proba on test
    
    fold_label_train = []
    fold_label_test = []
    seed_label_train = []
    seed_label_test = []
    subid_train = []
    subid_test = []
    best_param = {}

    # initialize result directory structures + filename patterns
    # Result_struct = init_result_dir(rdir) # done outside in the run.py
    featuredir = Result_struct.featuredir
    modeldir = Result_struct.modeldir
    preddir = Result_struct.preddir
    

    
    ######################################## start outer cv 

    # get outer_cv splits (stratifiedgroupkfold) while controlling for family structure (HCP!)
    outer_cv, groups = cv_control_for_family(subject_list, df_pheno, phenostr, seed, k_outer, df_family_info)


    # start outer_cv
    for foldind, (train_index, test_index) in enumerate(outer_cv):
        # print
        print('current fold: ', foldind+1)

        # train and test subject_id lists
        sub_train = subject_list[train_index]
        sub_test = subject_list[test_index]


        ######################################## get features and save dataframes
        # feature file names
        ftestfile = eval(Result_struct.featurefname_test)
        ftrainfile = eval(Result_struct.featurefname_train)

        # if exist, read in
        if os.path.isfile(ftestfile) and os.path.isfile(ftrainfile):
            print('Features already extracted and saved.')
            print('Test data path: ', ftestfile)
            print('Train data path: ', ftrainfile)
            test_data = pd.read_csv(ftestfile)
            train_data = pd.read_csv(ftrainfile)
            X_keys = [col for col in test_data.columns if 'PC' in col]
        else:
        # otherwise, get features - PC loadings and save for later use
            if feature_type == 'singlePC':
                F,Ft,Var,PCs,X_keys = perform_feature_extraction_singlePC(df_m, sub_train, sub_test, nroi, pcind, ntr, flip, clean, norm)
            elif feature_type == 'combinedPC':
                F,Ft,Var,PCs,X_keys = perform_feature_extraction_multiplePC(df_m, sub_train, sub_test, nroi, pcind, ntr, flip, clean, norm)
            else:
                print('please specify feature type: "singlePC" or "combinedPC"!')    
            train_data =get_dataframe(F,sub_train,df_pheno,phenostr,X_keys)
            test_data = get_dataframe(Ft,sub_test,df_pheno,phenostr,X_keys)
            test_data.to_csv(ftestfile, header=True,index=True)
            train_data.to_csv(ftrainfile, header=True,index=True)
            varfname = eval(Result_struct.varfname)  # save var
            epvar = pd.DataFrame(data=Var) # all rois for each PC
            epvar.to_csv(varfname)
        
        
        ######################################## settings for inner cv 
        y_keys = phenostr
        

        # Default: stratifiedkfold for inner loops, repeated 5 times; or simply run_cross_validation(cv=5)
        inner_cv = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=seed)

        # if stratifiedgroupkfold for inner loops; run_cross_validation(groups=groups)
        # inner_cv, groups = cv_control_for_family(sub_train, df_pheno, phenostr, seed, k_inner, df_family_info)
       
        
        
        ######################################## train model on training data using inner-cv settings

        # Note: this part could be replaced by sklearn functions if needed
        ##################################### Julearn specific ########################################################
        scores, estimator = run_cross_validation(
        X=X_keys, y=y_keys, data=train_data, preprocess_X=None,
        problem_type=J_model.probtype, model= J_model.name, model_params=J_model.model_params, return_estimator='final',
        cv=inner_cv, seed=seed)
         ###############################################################################################################

    
        ######################################## save the best parameters and models of each fold
        
        # when tuning hyperparameters: should has attribute "best_estimator_"
        if hasattr(estimator, 'best_estimator_'):
            model = estimator.best_estimator_
            bp = estimator.best_params_
            print('best parameter: ', bp)
            best_param = concatenate_best_params(best_param, bp, foldind)
        else:
            print('Has no attribute best_estimator_')
            model = estimator
    
        ######################################## save fitted model
        # save to fitted_models folder
        filename = eval(Result_struct.modelfname)
        pickle.dump(model, open(filename, 'wb'))

        ######################################## save predictions of each fold

        # predict on test data
        ytest_pred = model.predict(test_data[X_keys])
        ytest = test_data[y_keys]

        # final training predictions
        ytrain_pred = model.predict(train_data[X_keys])
        ytrain = train_data[y_keys]

        # get predict_proba on test for calculating roc_auc for binary classification if available
        ytest_proba = get_predproba_for_roc_auc(model, J_model, test_data, X_keys)

        # Print out some scores just out of curiosity
        if J_model.probtype == 'binary_classification':
            bacc_test = balanced_accuracy_score(ytest, ytest_pred)
            bacc_train = balanced_accuracy_score(ytrain, ytrain_pred)
            print('Test balanced accuracy for fold', foldind+1, 'is', bacc_test)
            print('Training balanced accuracy for fold', foldind+1, 'is', bacc_train)
        elif J_model.probtype == 'regression':
            pr_test = scipy.stats.pearsonr(ytest, ytest_pred)
            pr_train = scipy.stats.pearsonr(ytrain, ytrain_pred)
            print('Test pearson correlation for fold', foldind+1, 'is', pr_test[0])
            print('Training pearson correlation for fold', foldind+1, 'is', pr_train[0])

        ######################################## concatenate results across all folds
        pred_train = pred_train + list(ytrain_pred)
        true_train = true_train + list(ytrain)

        pred_test = pred_test + list(ytest_pred)
        true_test = true_test + list(ytest)
        proba_test = proba_test + list(ytest_proba)

        #fold_label_train = fold_label_train + [f'fold{foldind+1}']*len(sub_train)
        fold_label_train = fold_label_train + [foldind+1]*len(sub_train)
        fold_label_test = fold_label_test + [foldind+1]*len(sub_test)
        
        seed_label_train = seed_label_train + [seed]*len(sub_train)
        seed_label_test = seed_label_test + [seed]*len(sub_test)

        subid_train = subid_train + list(sub_train)
        subid_test = subid_test + list(sub_test)
        

    ######################################## save prediction results as dataframes for training and test separately of all folds   
    Ptest = []
    Ptest.append(pred_test)
    Ptest.append(true_test)
    df_pred_test = pd.DataFrame(Ptest).T
    df_pred_test.columns = ['pred','true']
    if proba_test: 
        df_pred_test['proba'] = proba_test
    df_pred_test.insert(0,'Subject',subid_test)
    df_pred_test.insert(0,'fold',fold_label_test)
    df_pred_test.insert(0,'seed',seed_label_test)

    Ptrain = []
    Ptrain.append(pred_train)
    Ptrain.append(true_train)
    df_pred_train = pd.DataFrame(Ptrain).T
    df_pred_train.columns = ['pred','true']
    df_pred_train.insert(0,'Subject',subid_train)
    df_pred_train.insert(0,'fold',fold_label_train)
    df_pred_train.insert(0,'seed',seed_label_train)
    
    # save to prediction folder
    fileptest = eval(Result_struct.predfname_test)
    df_pred_test.to_csv(fileptest, header=True)
    fileptrain = eval(Result_struct.predfname_train)
    df_pred_train.to_csv(fileptrain, header=True)

    ######################################## save best parameters of all folds
    BP = []
    parlist = list(best_param.keys())
    for p in parlist:
        BP.append(list(best_param[p]))
    df_best_params = pd.DataFrame(BP).T
    df_best_params.columns = parlist
    print('best paramters:')
    print(df_best_params)
    df_best_params.insert(0,'fold', np.arange(1,k_outer+1,1))
    df_best_params.insert(0,'seed',[seed] * k_outer)
    #print(df_best_params)

    # save to model folder
    filebp = eval(Result_struct.modelfname_bp)
    df_best_params.to_csv(filebp, header=True, index=True)

    #return df_pred_test, df_pred_train, df_best_params
    return df_pred_test, df_pred_train, df_best_params


def main_compute_prediction_scores(fmricondition, clfname, probtype, Result_struct, metric_list, seed_list, nfold, savepath, foldwise=1, test=True):
    """ compute prediction scores using specified measures for a given fmricondition+clfname """
    # fmricondition, clfname will be needed when loading and saving file
    # Result_struct: output of main_TOPF, gives the path to folders of prediction results or just Result_struct = init_result_dir(rdir)
    # metric_list = ['accuracy', 'balanced_accuracy', 'roc_auc'] for binary classification
    # metric_list = ['pearson','spearman','r2','mean_absolute_error'] for regression
    # metric_list should be given by user, but needs to be added to the func "cal_measures" beforehand
    # seed_list used
    # nfold: k_outer
    # foldwise = 1 : compute within each fold; 0/None: compute over all samples within each seed/rep
    # test=True: compute for predictions derived on test; otherwise: compute for predictions derived on training
    
    m_values = []
    m_labels = []
    fold_label = []
    seed_label = []
    mean_seed = []

    preddir = Result_struct.preddir
    if test:
        file = Result_struct.predfname_test
    else:
        file = Result_struct.predfname_train
    

    for seed in seed_list:
        
        df_test = pd.read_csv(eval(file))
        m_seed = []

        if foldwise: 
            print('Measures are computed within each fold')
            
            for fold in range(1,nfold+1,1):
                
                #df_temp = df_test[(df_test.seed==str(seed)) & (df_test.fold==str(fold))].reset_index(drop=True)
                df_temp = df_test[(df_test.seed==seed) & (df_test.fold==fold)].reset_index(drop=True)
                m_temp, mlist = cal_measures(df_temp, probtype, metric_list)  # calculate measures
                m_values = m_values + m_temp
                fold_label = fold_label + [fold]
                seed_label = seed_label + [seed]
                m_seed = m_seed + m_temp # within each seed

            # mean over all folds for a seed   
            m_seed = np.reshape(m_seed, [-1, len(mlist)]) 
            mean_seed = mean_seed + list(np.mean(m_seed, axis = 0))

        else:
            print('Measures are computed within each seed over all samples')
            df_temp = df_test[df_test.seed==seed].reset_index(drop=True)
            m_temp, mlist = cal_measures(df_temp, probtype, metric_list)  # calculate measures
            m_values = m_values + m_temp
            seed_label = seed_label + [seed]
            fold_label = []
            mean_seed = m_values 

    m_labels = mlist

    # save as dataframe
    m_values = np.array(m_values)
    m_values = np.reshape(m_values, [-1, len(m_labels)])
    df_measure = pd.DataFrame(data=m_values,columns=m_labels)
    if fold_label:
        df_measure.insert(0,'fold', fold_label)
    df_measure.insert(0,'seed', seed_label)

    # print results
    mean_all = np.mean(m_values, axis = 0)
    print('The measures computed are: ', m_labels)
    print('The mean over all seeds are: ', mean_all)

    mean_seed = np.reshape(mean_seed, [-1, len(m_labels)])
    # save
    if savepath:
        df_measure.to_csv(savepath, header=True)

    return mean_all, mean_seed, df_measure, m_labels



############################################# END #############################################
