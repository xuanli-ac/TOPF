# Topography-based predictive framework (TOPF)
>Code for implementing TOPF and relevant analysis

## Description
* data - gives the IDs of the HCP subjects included in the dataset1 and dataset2 (described in the manuscript)
* data/ExampleData - includes the behavioural scores of all subjects (Behaviour.csv; available from HCP) and fMRI time series of two subjects as an example (raw data available from HCP). Family structure information should be accessed via HCP after application.
* preproc - Custom bash and Matlab code for preprocessing imaging data
* preproc/GLM - Custom code for preprocessing GLM-derived activation maps downloaded from HCP

* code/TOPF.py - for performing TOPF on terminal (input arguments in order: datatype clipind phenostr permind clfname prob_type flag_coef seed pcind
* code/TOPF_wrapper.sh + TOPF_submit.sh - if use cluster (give input arguments in *_submit.sh)
* code/cal_featureimportance.py - compute the feature importance of each ROI for a given CV split

* ** The code folder (containing code to implement TOPF and relevant analysis) will be made available upon manuscript acceptance. 

## Usage example:

#### TOPF 

on terminal:

``python TOPF.py movie 1 PMAT24_A_CR 0 ridge regression 0 2 1``

datatype = movie, clipind = 1, phenostr = PMAT24_A_CR, permind = 0 (no perm), clfname = ridge, prob_type = regression, flag_coef = 0 (don't save regression weights), seed = 2 (cv splits seed), pcind = 1 (PC1 loadings as features)

or

``./TOPF.sh movie 1 PMAT24_A_CR 0 ridge regression 0 2 1``

on cluster (with htcondor):

``./TOPF_submit.sh | condor_submit``

#### Compute feature importance 

on terminal:

``python cal_featureimportance.py movie 1 PMAT24_A_CR 0 ridge 100 66``

datatype = movie, clipind = 1, phenostr = PMAT24_A_CR, cvind = 0 (0 - 9 cv repeats), clfname = ridge, nrep = 100, seed = 66 (permutation seed)

or

``./cal_featureimportance.sh movie 1 PMAT24_A_CR 0 ridge 100 66``

on cluster:

``./cal_featureimportance.sh | condor_submit``
