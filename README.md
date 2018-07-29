Automated Preprocessing Pipeline for Fetal Resting-State fMRI

1. Introduction
Recent advancements in resting-state functional magnetic resonance (rs-fMRI) imaging have enabled observation of the human brain in utero, a period of rapid growth and development previously inaccessible. The application of fMRI to studying the fetal brain has produced unique challenges and is an opportune field for methods development. In this study, we present a novel application of a Convolutional Neural Network (CNN) to a challenging image classification task: identifying the fetal brain. Resting-state fMRI data was obtained from 197 fetuses (gestational age 24-39 weeks, M=30.9, SD=4.2). The output from automated brain tissue identification is compared with the ground truth of 1,168 manually drawn brain masks. We report that automated fetal brain classification is achievable at the same integrity of manual methods, in a fraction of the time. There is a 92% spatial overlap between automated and manual fetal brain masks in a held-out test set of 48 subjects and each auto-mask is generated in approximately 2.5 seconds compared to several hours for an expert working manually.  Furthermore, we unite the automated brain masking model with an adapted realignment technique to better handle motion, reorientation, and normalize to age-specific fetal templates to create the first open source, standardized, fully automated preprocessing pipeline for fetal functional MRI data.

1.1 Installation & Requirements
Required libraries:
NumPy
TensorFlow
MedPy

1.2 Necessary data prep steps
Input data must be of dimensions 96 x 96. The data used in training was resampled to voxel size 3.5 mm^3, then zero padded to 96 x 96 x 37. 
See lines 4-26 of FullFetalPreprocessPipeline.sh for an example of preprocessing commands.

2. Running the auto-mask code options

2.1 Use pre-trained model
Images should be in 3D volume format (split the 4D time series into individual volumes). File naming should be consistent. Currently, the code expects images to be in a folder called "images/" and named as "zpr_SubjectID_runID_vol0000.nii". 
CreateMask.py is the code to run to create new masks. 

2.2 Train model on your data
Instructions for training model on new data coming soon. 


