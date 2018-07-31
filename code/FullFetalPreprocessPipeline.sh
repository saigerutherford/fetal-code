#!/bin/bash

# Part 1: Preprocess for automasking
# Run from within preproc directory
cd preproc/
for sub in `cat ../sublist` #sublist is a text file of 4D NIFTI files (1 filename per line)
do
#Delete orientation, resample, zeropad, split 4D time series
fslorient -deleteorient ${sub}
fslsplit ${sub} ${sub}_vol -t
#Unzip files
for file in *.nii.gz
do
	gunzip ${file}
done
#Check dimensions, resample if not 96 x 96
dim1=`fslinfo ${sub} | grep -w "dim1"` #RL dim
dim2=`fslinfo ${sub} | grep -w "dim2"` #AP dim
if
dim1 ! == 96 && dim2 ! == 96
	then
		for vol in ${sub}_vol*
			   do
		3dresample -dxyz 3.5 3.5 3.5 -prefix r_${sub}_${vol} -input ${sub}_${vol}
		3dZeropad -RL 96 -AP 96 -prefix zpr_${sub}_${vol} r_${sub}_${vol}
		done
fi
done
mv zpr_* ../images/

# Part 2: Run automask code, quality check masks (Pass/Fail)
# createMask.py expects there to be a directory called "images/" where the 96 x 96 sized files live,
# and that they are named zpr_SubjectID_runID.nii
source /home/slab/environments/tensorflow/bin/activate
python createMask.py

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

#Quality check 4D cluster image
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

#Quality Check
fslview e_4D_${i}

# Part 3: Other preprocessing steps (realign, motion denoise, normalize)
#Realign
fslsplit e_4D_${sub}.nii.gz e_${sub}_vol -t
for vol in e_${sub}_vol*
	do
		flirt -in ${vol} -out re_${vol} -mats -plots
	done

#Create confounds file
fsl_motion_outliers -i red_4D_${i} -o confounds_${i} --nomoco -s rp_$i} -p motion_plot_${i} --dvars

#Normalize
for vol in re_${sub}_vol*
	do
		flirt -in ${vol} -ref ${template} -out wre_${vol}
	done

#Merge normalized file
fslmerge -t t_* 4D_t_${sub}

#View final product
fslview 4D_t_${sub}
