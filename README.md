# Automated Brain Masking of Fetal Functional MRI Data
**Preprint:** https://www.biorxiv.org/content/early/2019/01/21/525386

**Abstract:** Fetal resting-state functional magnetic resonance imaging (rs-fMRI) has emerged as a critical new approach for characterizing brain network development before birth. Despite rapid and widespread growth of this approach, at present we lack neuroimaging processing pipelines suited to address the unique challenges inherent in this data type. Here, we solve the most challenging processing step, rapid and accurate isolation of the fetal brain from surrounding tissue across thousands of non-stationary 3D brain images. Leveraging our library of 1,255 manually traced fetal fMRI images from 207 subjects, we trained a Convolutional Neural Network (CNN) that achieved excellent accuracy (>90%) across two held-out test sets from separate populations. Furthermore, we unite the auto-masking model with additional fMRI preprocessing steps from existing softwares and provide insight into our adaptation of each step. This work represents an initial stepping stone towards a fully comprehensive, open source pipeline for fetal functional MRI data preprocessing. 

Primary data used for pipeline development were acquired at Wayne State
University School of Medicine during the course of projects supported by National
Institutes of Health (NIH) awards MH110793 and ES026022.

For access to raw fetal functional time-series data used in the development of this code please contact Moriah Thomason Moriah.Thomason@nyulangone.org

### Repository organization
**checkpoints -->** contains the saved models. **2018-06-07_14:07** is the model trained using _**train, validation, and test split**_ (129, 20, 48 subjects; 855, 102, 211 volumes) **2018-06-08_10:47** is the model trained on _**all**_ labeled data.

**code -->** this directiory contains all necessary scripts for running the pretrained model (`createMasks.py`), or training your own model (`buildModel.py` and `trainModel.py`). PatientMetrics.csv contains the evaluation info for all subjects/volumes within the WSU test set. code/FullFetalPreprocessPipeline.sh --> Example pipeline using auto-mask and FSL. **Work in progress, not fully tested or setup without hard-coded paths**.


**figures -->** Jupyter notebook used to make the figures in the manuscript. BySubjectEvaluation.csv contains the WSU test set evaluation metrics grouped by subject and reported as an average (one test set subject has several volumes). ByVolumeEvaluation.csv contains the WSU test set evaluation metrics reported per volume. Yale_PatientMetrics3.csv contains the Yale test set evaluation metrics.

**summaries -->** Contains the summaries for both models described above (in the checkpoints directory description) that can be viewed using tensorboard.
`tensorboard --logdir=summaries/model_name`

### Installation & Requirements
Required libraries: See Requirements.txt file

### Necessary data prep steps (prior to auto-masking)
Input data must be of dimensions 96 x 96 x N. The data used in training was resampled to voxel size 3.5 mm^3, then zero padded to 96 x 96 x 37. See lines 4-26 of FullFetalPreprocessPipeline.sh for an example of preprocessing commands.

## Running the auto-mask code options
### Use pre-trained model
Images should be in 3D volume format (split the 4D time series into individual volumes). File naming should be consistent. Currently, the code expects images to be in a folder called "images/" and named as "zpr_SubjectID_runID_vol0000.nii".
createMask.py is the script to run to create new masks using the model trained with all 1,168 hand drawn brain masks.

### Train model on your data
Instructions for training model on new data coming soon.

## Other preprocessing steps
1. Merge auto-masks into 4D file to view as overlay on data in order to quality check the masks.
2. Cluster and binarize the probability masks
3. Quality check binarized mask
4. Resample mask back into subject's native space
5. Apply binarized native space brain mask to raw data to extract the fetal brain and discard other tissues
6. Quality check
7. Realign time series and identify usable low motion volumes
8. Quality Check
9. Normalize to age-matched fetal template
10. Quality check
