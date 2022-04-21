#!/bin/bash
# v3.2

logs_dir=/data/project/TOPF/preproc/logs_newmask
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

# print the .submit header
printf "# The environment

universe              = vanilla
getenv                = True
request_cpus          = 1
request_memory        = 10G

# Execution
initialdir            = /data/project/TOPF/preproc
executable            = /data/project/TOPF/preproc/3Thcp_preproc_para.sh
transfer_executable   = False
\n"

# Create a Job for each parameter
data=tfMRI_MOTOR_RL
subid=/data/project/TOPF/preproc/179Movie/subid.txt
cfg=/data/project/TOPF/preproc/DPARSFA_C.mat
roi=/data/project/TOPF/preproc/shen_2mm_268_parcellation_mni.nii

for sub in $(cat $subid); do
	printf "arguments   = ${data} ${sub} ${cfg} ${roi}\n"
	printf "requirements = Machine == \"cpu23.htc.inm7.de\"\n"
	#printf "requirements = Machine == \"cpu11.htc.inm7.de\" || Machine == \"cpu12.htc.inm7.de\"\n"
	printf "log         = ${logs_dir}/\$(Cluster).\$(Process)_sub${sub}.log\n"
	printf "output      = ${logs_dir}/\$(Cluster).\$(Process)_sub${sub}.out\n"
	printf "error       = ${logs_dir}/\$(Cluster).\$(Process)_sub${sub}.err\n"
	printf "Queue\n\n"
done
