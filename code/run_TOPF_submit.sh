#!/bin/bash
# v3.2

###### HTCondor/Juseless variables ###################
init_dir='/TOPF'
code_dir='code'
name_py='run_TOPF.py'
name_wrapper='run_TOPF.sh'
######################################################


# print the .submit header
printf "# The environment

universe              = vanilla
getenv                = True
request_cpus          = 1
request_memory        = 5G

# Execution
initialdir            = ${init_dir}/${code_dir}
executable            = ${init_dir}/${code_dir}/${name_wrapper}
transfer_executable   = False
transfer_input_files = ${name_py}
\n"


############################################### set up arguments for TOPF here ##############################################
# project dir
wkdir=${init_dir}
sublist=${init_dir}/data/SubjectID_Dataset2_HCP7T_179.txt
rpath=${init_dir}/results
settingpath=${init_dir}/${code_dir}/ridge_settings.txt

# other settings
nroi=268 # 268
seed=0
condind=1  # 1-7
cutntr=0 # cutntr<=0: using full length; otherwise: cut to the given length e.g., ntr=170
featuretype='singlePC' # 'singlePC' or combinedPC
pcind=1 # which PC or to the largest PC index when combinedPC
threshold=0
clfname='ridge' # e.g., 'svm', which classifier (see all availabel classifiers here: https://juaml.github.io/julearn/main/steps.html)

seedlist=(0 2 10 12 15 20 25 36 38 43)
#phenolist=(PMAT24_A_CR ListSort_Unadj ER40_CR NEOFAC_A NEOFAC_O NEOFAC_C NEOFAC_N NEOFAC_E)
#threlist=(0.05 0.07 0.1)
#threlist=(0 0.02 0.03 0.05 0.1)


## for test
threlist=(0)
phenolist=(ListSort_Unadj)
seedlist=(0)
# nroi=5

####################################################################################################################################

# Create a Job for each parameter combination
for threshold in "${threlist[@]}"; do
    for phenostr in "${phenolist[@]}"; do
        ############################################### set up log folder ############################################
        logs_dir=${init_dir}/${code_dir}/logs_${phenostr}_thre${threshold}_${cutntr}_${featuretype}_${pcind}
        # create the logs dir if it doesn't exist
        [ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"
        ##############################################################################################################
        for seed in "${seedlist[@]}"; do
	        for condind in {1..7}; do
		        printf "arguments   = ${wkdir} ${sublist} ${rpath} ${nroi} ${seed} ${condind} ${phenostr} ${cutntr} ${featuretype} ${pcind} ${threshold} ${clfname} ${settingpath}\n"
		        #printf "requirements = Machine == \"cpu23.htc.inm7.de\"\n"
		        #printf "requirements = Machine == \"cpu11.htc.inm7.de\" || Machine == \"cpu12.htc.inm7.de\"\n"
		        printf "log         = ${logs_dir}/\$(Cluster).\$(Process)_${phenostr}_${condind}_seed${seed}.log\n"
		        printf "output      = ${logs_dir}/\$(Cluster).\$(Process)_${phenostr}_${condind}_seed${seed}.out\n"
		        printf "error       = ${logs_dir}/\$(Cluster).\$(Process)_${phenostr}_${condind}_seed${seed}.err\n"
		        printf "Queue\n\n" 
            done
        done   
	done
done
