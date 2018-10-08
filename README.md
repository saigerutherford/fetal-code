# Automated Preprocessing Pipeline for Fetal fMRI Data
## Introduction
Recent advancements in resting-state functional magnetic resonance (rs-fMRI) imaging have enabled observation of the human brain in utero, a period of rapid growth and development previously inaccessible. The application of fMRI to studying the fetal brain has produced unique challenges and is an opportune field for methods development. In this study, we present a novel application of a Convolutional Neural Network (CNN) to a challenging image classification task: identifying the fetal brain. Resting-state fMRI data was obtained from 197 fetuses (gestational age 24-39 weeks, M=30.9, SD=4.2). The output from automated brain tissue identification is compared with the ground truth of 1,168 manually drawn brain masks. We report that automated fetal brain classification is achievable at the same integrity of manual methods, in a fraction of the time. There is a 92% spatial overlap between automated and manual fetal brain masks in a held-out test set of 48 subjects and each auto-mask is generated in approximately 2.5 seconds compared to several hours for an expert working manually.  Furthermore, we unite the automated brain masking model with an adapted realignment technique to better handle motion, reorientation, and normalize to age-specific fetal templates to create the first open source, standardized, fully automated preprocessing pipeline for fetal functional MRI data.
[Slides from E-Poster at ISMRM 2018](https://www.slideshare.net/SaigeRutherford/ismrm-2018-eposter)
### Repository organization
Checkpoints --> contains the saved models. **2018-06-07_14:07** is the model trained using _**train, validation, and test split**_ (129, 20, 48 subjects; 855, 102, 211 volumes) **2018-06-08_10:47** is the model trained on _**all**_ labeled data. 

Summaries --> Contains the summaries for both models described above that can be viewed using tensorboard. 
`tensorboard --logdir=summaries/model_name`

Code --> this directiory contains all necessary scripts for running the pretrained model (`createMasks.py`), or training your own model (`buildModel.py` and `trainModel.py`). PatientMetrics.csv contains the evaluation info for all subjects/volumes within the test set.

FullFetalPreprocessPipeline.sh --> Example pipeline using auto-mask and FSL. **Work in progress, not a complete pipeline or tested**. 
### Installation & Requirements
Required libraries:
1. NumPy
2. TensorFlow
3. MedPy
### Necessary data prep steps (prior to auto-masking)
Input data must be of dimensions 96 x 96 x n. The data used in training was resampled to voxel size 3.5 mm^3, then zero padded to 96 x 96 x 37. See lines 4-26 of FullFetalPreprocessPipeline.sh for an example of preprocessing commands.
## Running the auto-mask code options
### Use pre-trained model
Images should be in 3D volume format (split the 4D time series into individual volumes). File naming should be consistent. Currently, the code expects images to be in a folder called "images/" and named as "zpr_SubjectID_runID_vol0000.nii". 
createMask.py is the script to run to create new masks using the model trained with all 1,168 hand drawn brain masks. 
### Train model on your data
Instructions for training model on new data coming soon. 
## Other preprocessing steps
1. Merge auto-masks into 4D file to view as overlay on data in order to quality check the masks. 
2. Cluster and binarize the probability mask 
3. Quality check binarized mask
4. Resample mask back into subject's native space
5. Apply binarized native space brain mask to raw data to extract the fetal brain and discard other tissues
6. Quality check
7. Realign time series and identify usable low motion volumes
8. Quality Check
9. Normalize to age-matched fetal template
10. Quality check

