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
