#setup fsl
source /etc/fsl/fsl.sh

# for SOCIAL
for line in $(cat /data/project/TOPF/GLM_HCP/subid_100un.txt);
do 
	cd /data/project/TOPF/hcp-openaccess/HCP1200/
	for ncope in {1..2};
	do
    	datalad get ${line}/MNINonLinear/Results/tfMRI_SOCIAL/tfMRI_SOCIAL_hp200_s4_level2vol.feat/cope${ncope}.feat/stats/zstat1.nii.gz
    done
    wkd=/data/project/TOPF/hcp-openaccess/HCP1200/${line}/MNINonLinear/Results/tfMRI_SOCIAL/tfMRI_SOCIAL_hp200_s4_level2vol.feat/
    
    cd /data/project/TOPF/GLM_HCP/social_abs
    mkdir ${line}
    cd ${line}
    fslmaths ${wkd}/cope1.feat/stats/zstat1.nii.gz -abs abs_cope1.nii.gz
    fslmaths ${wkd}/cope2.feat/stats/zstat1.nii.gz -abs abs_cope2.nii.gz
    fslmaths abs_cope1.nii.gz -max abs_cope2.nii.gz abs_max.nii
    
    rm -rf abs_cope1.nii.gz
    rm -rf abs_cope2.nii.gz
    cd ${wkd}
    datalad drop ${wkd}
done