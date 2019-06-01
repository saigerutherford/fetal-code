#Concatenate masks, view as overlay on raw time series
for sub in `cat sublist`; do fslmerge -t 4Dmask_${sub} pred_${sub}_vol*.nii; echo ${sub}; done
for sub in `cat sublist`; do fslmerge -t 4D_${sub} ${sub}_vol*.nii; echo ${sub}; done
#Quality check
for sub in 4D_*; do mask_name=`echo ${sub//4D_}`; fslview ${sub} 4Dmask_${mask_name} "Copper" -t 0.5
#keep biggest cluster & binarize the probability masks (needs to be done on 3D volumes, not on 4D timeseries)
for mask in pred_*
    do 
        3dclust -1clip 0.1 -NN3 -savemask c_${mask}.nii ${mask}
        fslmaths c_${mask}.nii -uthr 1 -bin bc_${mask}
    done
#Quality check 4D cluster image (need to add loop here)
fslmerge -t 4Dbc_${sub} bc_${sub}
fslview 4Dbc_${sub}.nii.gz
#Undo ZeroPad & Resample into native space
#First check native space dimensions
fslinfo ${sub} | grep -w "dim1" #RL dim
fslinfo ${sub} | grep -w "dim2" #AP dim
3dZeropad -R -10 -L -10 -A -10 -P -10 -I -5 -prefix nm_${sub}.nii 4D_brain_cluster_${sub}.nii.gz
3dresample -master 4D_${sub}.nii -infile native_mask_${sub}.nii -prefix r_native_mask_${sub}.nii -rmode NN
#View native space mask overlay
fslview r_native_mask_${sub}.nii 4D_${sub}.nii
#Swap dimensions if needed
fslswapdim -x/x -y/y -z/z r_native_mask_${sub}.nii fr_native_mask_${sub}
#Apply masks to data
fslmaths 4D_${sub}.nii -mul frnm_${sub}.nii e_4D_${sub} 