# copy code and run in command window

# hcp
export DATALAD_hcp_s3_key_id=keyid
export DATALAD_hcp_s3_secret_id=password

#setup fsl
source /etc/fsl/fsl.sh

# for motor 
for line in $(cat /data/project/TOPF/preproc/100Unrelated/subid.txt);
do 
	cd /data/project/TOPF/hcp-openaccess/HCP1200/
	for ncope in {1..7};
	do
    #datalad get ${line}/MNINonLinear/Results/tfMRI_MOTOR_LR/tfMRI_MOTOR_LR.nii.gz
    	datalad get ${line}/MNINonLinear/Results/tfMRI_MOTOR/tfMRI_MOTOR_hp200_s4_level2vol.feat/cope${ncope}.feat/stats/zstat1.nii.gz
    done
    wkd=/data/project/TOPF/hcp-openaccess/HCP1200/${line}/MNINonLinear/Results/tfMRI_MOTOR/tfMRI_MOTOR_hp200_s4_level2vol.feat/
    
    cd /data/project/TOPF/GLM_HCP
    mkdir ${line}
    cd ${line}
    fslmaths ${wkd}/cope1.feat/stats/zstat1.nii.gz -abs abs_cope1.nii.gz
    fslmaths ${wkd}/cope2.feat/stats/zstat1.nii.gz -abs abs_cope2.nii.gz
    fslmaths ${wkd}/cope3.feat/stats/zstat1.nii.gz -abs abs_cope3.nii.gz
    fslmaths ${wkd}/cope4.feat/stats/zstat1.nii.gz -abs abs_cope4.nii.gz
    fslmaths ${wkd}/cope5.feat/stats/zstat1.nii.gz -abs abs_cope5.nii.gz
    fslmaths ${wkd}/cope6.feat/stats/zstat1.nii.gz -abs abs_cope6.nii.gz

    fslmaths abs_cope1.nii.gz -max abs_cope2.nii.gz -max abs_cope3.nii.gz -max abs_cope4.nii.gz -max abs_cope5.nii.gz -max abs_cope6.nii.gz abs_max.nii

    cd ${wkd}
    datalad drop ${wkd}
done




