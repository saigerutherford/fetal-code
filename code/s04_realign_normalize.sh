# Part 3: Other preprocessing steps (realign, motion denoise, normalize)
sublist=$1
path=$2
template=$3 #correct age-matched fetal template 

#Realign
for sub in sublist; do fslsplit e_4D_${sub}.nii.gz e_${sub}_vol -t; done
for vol in e_${sub}_vol*; do flirt -in ${vol} -out re_${vol} -mats -plots; done

#Create confounds file
fsl_motion_outliers -i red_4D_${i} -o confounds_${i} --nomoco -s rp_$i} -p motion_plot_${i}

#Normalize
for vol in re_${sub}_vol*; do flirt -in ${vol} -ref ${template} -out wre_${vol}; done
#Merge normalized file
fslmerge -t t_* 4D_t_${sub}